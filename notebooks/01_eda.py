"""EDA публичного среза KION для подборки «Вам понравится».

Запуск из корня репозитория:

    python notebooks/01_eda.py

Нужны pandas, numpy и matplotlib (группа зависимостей eda в pyproject.toml).
Csv лежат в data/data_original/ и в git не входят.
Скрипт пишет графики в reports/figures и сводку в reports/eda_summary.json.
Выводы — в reports/eda.md, ноутбук с теми же рисунками — notebooks/01_eda.ipynb.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "data_original"
ZIP_PATH = ROOT / "data" / "data_original.zip"
FIG = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "eda_summary.json"

ZIP_SHA256 = "50fc5b7db78cbda4c3d0ee5bb5bea4c416cff41e8af58aab0b661d0d3e9dfa9b"
ZIP_MD5 = "87ec2729ca79a866f0d5b6bd2e94e5fa"

PAPER_INTERACTIONS = 5_476_251
PAPER_USERS = 962_179
PAPER_ITEMS = 15_706

CANDIDATES = (
    {"name": "pct50", "pct": 50, "dur_min": 0},
    {"name": "pct50_5min", "pct": 50, "dur_min": 5},
    {"name": "pct50_10min", "pct": 50, "dur_min": 10},
    {"name": "pct50_15min", "pct": 50, "dur_min": 15},
    {"name": "pct30_10min", "pct": 30, "dur_min": 10},
    {"name": "pct80_10min", "pct": 80, "dur_min": 10},
)
LIFT_CANDIDATE = "pct50_10min"
PCT_LEVELS = (0, 10, 30, 50, 70, 90, 100)
DUR_LEVELS = (0, 1, 3, 5, 10, 15, 20, 30)
HEAD_KS = (10, 20, 50, 100, 200, 500, 1000, 2000)
LIFT_MIN_COUNTS = (1, 2, 3, 5, 10, 20)
LIFT_HISTORY_CAP = 40
LIFT_ITEM_SAMPLE = 800
LIFT_MIN_POP = 30
RANKING_LIMIT = 250
K = 10

PRIMARY = "#1f4e79"
SECOND = "#d06a2b"
THIRD = "#3d7a5a"
NEUTRAL = "#5c6670"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def file_hash(path: Path, algo: str) -> str:
    digest = hashlib.new(algo)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_blank(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.eq("") | text.str.lower().isin(["nan", "none", "<na>"])


def split_genres(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def ranking_frame(counts: pd.Series) -> pd.DataFrame:
    frame = counts.rename("cnt").reset_index()
    if list(frame.columns) != ["item_id", "cnt"]:
        frame.columns = ["item_id", "cnt"]
    return frame.sort_values(["cnt", "item_id"], ascending=[False, True], kind="mergesort")


def ranking_ids(counts: pd.Series, limit: int = RANKING_LIMIT) -> list[int]:
    if counts.empty:
        return []
    ordered = ranking_frame(counts)
    return [int(item_id) for item_id in ordered["item_id"].head(limit).tolist()]


def build_rel(frame: pd.DataFrame) -> dict[int, set[int]]:
    relevant: dict[int, set[int]] = {}
    if frame.empty:
        return relevant
    for user_id, item_id in zip(
        frame["user_id"].to_numpy(), frame["item_id"].to_numpy(), strict=True
    ):
        relevant.setdefault(int(user_id), set()).add(int(item_id))
    return relevant


def recommend(ranking: list[int], seen: set[int] | None, k: int = K) -> list[int]:
    chosen: list[int] = []
    blocked = seen or set()
    for item_id in ranking:
        if item_id in blocked:
            continue
        chosen.append(item_id)
        if len(chosen) == k:
            break
    return chosen


def recommend_with_backfill(
    primary: list[int],
    fallback: list[int],
    seen: set[int] | None,
    k: int = K,
) -> tuple[list[int], int]:
    chosen: list[int] = []
    blocked = set(seen or ())
    for item_id in primary:
        if item_id in blocked or item_id in chosen:
            continue
        chosen.append(item_id)
        if len(chosen) == k:
            return chosen, 0
    backfill = 0
    for item_id in fallback:
        if item_id in blocked or item_id in chosen:
            continue
        chosen.append(item_id)
        backfill += 1
        if len(chosen) == k:
            break
    return chosen, backfill


LOG_DISCOUNT = [1.0 / math.log2(rank + 1) for rank in range(1, K + 1)]
IDCG = [0.0]
for discount in LOG_DISCOUNT:
    IDCG.append(IDCG[-1] + discount)


def score_list(
    recommended: list[int], relevant: set[int], k: int = K
) -> tuple[float, float, float]:
    hits = 0
    dcg = 0.0
    average_precision = 0.0
    for index, item_id in enumerate(recommended[:k]):
        if item_id not in relevant:
            continue
        hits += 1
        dcg += LOG_DISCOUNT[index]
        average_precision += hits / (index + 1)
    recall = hits / len(relevant)
    ndcg = dcg / IDCG[min(k, len(relevant))]
    if hits:
        average_precision /= min(len(relevant), k)
    return ndcg, recall, average_precision


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def metric_bundle(
    user_ids: list[int],
    relevant: dict[int, set[int]],
    seen: dict[int, set[int]],
    ranking: list[int],
    group_rankings: dict[tuple[str, int], list[int]] | None = None,
    demo_map: dict[int, tuple[str, int]] | None = None,
    tail_items: set[int] | None = None,
) -> dict:
    ndcg_values: list[float] = []
    recall_values: list[float] = []
    map_values: list[float] = []
    tail_ndcg: list[float] = []
    tail_recall: list[float] = []
    covered: set[int] = set()
    same_as_global = 0
    backfill_slots = 0
    users_with_backfill = 0
    users_without_demo = 0
    global_head = ranking[:K]
    for user_id in user_ids:
        rel = relevant[user_id]
        blocked = seen.get(user_id)
        backfill = 0
        if group_rankings is None:
            recommended = recommend(ranking, blocked)
        else:
            key = None if demo_map is None else demo_map.get(user_id)
            primary = group_rankings.get(key) if key is not None else None
            if not primary:
                users_without_demo += 1
                recommended = recommend(ranking, blocked)
                backfill = len(recommended)
            else:
                recommended, backfill = recommend_with_backfill(primary, ranking, blocked)
        if backfill:
            users_with_backfill += 1
            backfill_slots += backfill
        covered.update(recommended)
        if recommended == global_head:
            same_as_global += 1
        ndcg, recall, average_precision = score_list(recommended, rel)
        ndcg_values.append(ndcg)
        recall_values.append(recall)
        map_values.append(average_precision)
        if tail_items is not None:
            tail_rel = rel & tail_items
            if tail_rel:
                tail_n, tail_r, _ = score_list(recommended, tail_rel)
                tail_ndcg.append(tail_n)
                tail_recall.append(tail_r)
    n_users = len(user_ids)
    return {
        "n_users": n_users,
        "ndcg@10": summarize(ndcg_values),
        "recall@10": summarize(recall_values),
        "map@10": summarize(map_values),
        "hitrate@10": float(np.mean(np.asarray(ndcg_values) > 0)) if ndcg_values else None,
        "coverage_items": len(covered),
        "share_lists_equal_global_top10": (same_as_global / n_users) if n_users else None,
        "share_users_without_demo": (users_without_demo / n_users) if n_users else None,
        "share_users_with_backfill": (users_with_backfill / n_users) if n_users else None,
        "mean_backfill_slots": (backfill_slots / n_users) if n_users else None,
        "tail_outside_top100": {
            "n_users": len(tail_ndcg),
            "ndcg@10": summarize(tail_ndcg)["mean"],
            "recall@10": summarize(tail_recall)["mean"],
        },
    }


def group_rankings_for(
    interactions: pd.DataFrame,
    positive: pd.Series,
    demo: pd.DataFrame,
) -> dict[tuple[str, int], list[int]]:
    rows = interactions.loc[positive, ["user_id", "item_id"]]
    merged = rows.merge(demo, on="user_id", how="inner")
    if merged.empty:
        return {}
    counts = merged.groupby(["age", "kids_flg", "item_id"], sort=False).size()
    rankings: dict[tuple[str, int], list[int]] = {}
    for (age, kids_flg), group_counts in counts.groupby(level=[0, 1], sort=False):
        item_counts = group_counts.droplevel([0, 1])
        rankings[(str(age), int(kids_flg))] = ranking_ids(item_counts)
    return rankings


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e8eb", zorder=0)
    ax.set_axisbelow(True)


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / name, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def require_inputs() -> dict:
    missing = [
        name
        for name in ("interactions.csv", "users.csv", "items.csv")
        if not (DATA / name).exists()
    ]
    if missing:
        raise SystemExit(
            "Не найдены csv в data/data_original/: "
            + ", ".join(missing)
            + ". Скачайте data_original.zip и распакуйте его в data/."
        )
    hashes = {}
    if ZIP_PATH.exists():
        hashes = {
            "sha256": file_hash(ZIP_PATH, "sha256"),
            "md5": file_hash(ZIP_PATH, "md5"),
        }
        if hashes["sha256"] != ZIP_SHA256 or hashes["md5"] != ZIP_MD5:
            raise SystemExit(f"Хеш архива не совпал с зафиксированным: {hashes}")
    return hashes


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("читаю csv")
    interactions = pd.read_csv(DATA / "interactions.csv", parse_dates=["last_watch_dt"])
    users = pd.read_csv(DATA / "users.csv")
    items = pd.read_csv(DATA / "items.csv")
    interactions["user_id"] = interactions["user_id"].astype("int64")
    interactions["item_id"] = interactions["item_id"].astype("int64")
    interactions["total_dur"] = interactions["total_dur"].astype("int64")
    interactions["watched_pct"] = interactions["watched_pct"].astype("float64")
    users["user_id"] = users["user_id"].astype("int64")
    items["item_id"] = items["item_id"].astype("int64")
    for column in ("age", "income", "sex"):
        users[column] = users[column].astype("string").str.strip()
        users.loc[users[column].isin(["", "<NA>", "nan", "None"]), column] = pd.NA
    users["kids_flg"] = pd.to_numeric(users["kids_flg"], errors="coerce").astype("Int64")
    items["content_type"] = items["content_type"].astype("string").str.strip()
    return interactions, users, items


def value_counts(series: pd.Series, limit: int = 20) -> dict[str, int]:
    counts = series.fillna("(пусто)").astype(str).value_counts()
    head = {str(key): int(value) for key, value in counts.head(limit).items()}
    if len(counts) > limit:
        head["(прочие)"] = int(counts.iloc[limit:].sum())
    return head


def _duration_summary(interactions: pd.DataFrame, content: pd.Series, pct: pd.Series) -> dict:
    minutes = interactions["total_dur"] / 60.0
    seconds = interactions["total_dur"]

    def quantiles(mask: pd.Series) -> dict:
        sample = minutes.loc[mask]
        if sample.empty:
            return {"n": 0}
        points = sample.quantile([0.1, 0.5, 0.9])
        return {
            "n": int(mask.sum()),
            "p10": float(points.loc[0.1]),
            "p50": float(points.loc[0.5]),
            "p90": float(points.loc[0.9]),
        }

    high = pct >= 50
    return {
        "max_hours": float(seconds.max() / 3600),
        "n_ge_24h": int((seconds >= 24 * 3600).sum()),
        "n_ge_100h": int((seconds >= 100 * 3600).sum()),
        "n_ge_1000h": int((seconds >= 1000 * 3600).sum()),
        "p99_minutes": float(minutes.quantile(0.99)),
        "all": quantiles(pct.notna()),
        "positive_all": quantiles(high),
        "positive_film": quantiles(high & content.eq("film")),
        "positive_series": quantiles(high & content.eq("series")),
        "pct_ge_50_and_under_10_min": int((high & (minutes < 10)).sum()),
    }


def profile(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
) -> dict:
    log("профиль качества")
    inter_users = set(interactions["user_id"].unique().tolist())
    inter_items = set(interactions["item_id"].unique().tolist())
    user_ids = set(users["user_id"].tolist())
    item_ids = set(items["item_id"].tolist())
    history = interactions.groupby("user_id").size()
    daily = interactions.groupby("last_watch_dt").size().sort_index()
    span = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    missing_days = [day.strftime("%Y-%m-%d") for day in span.difference(daily.index)]
    last_day = int(daily.iloc[-1])
    previous = daily.iloc[:-1]
    content_map = items.set_index("item_id")["content_type"]
    content = interactions["item_id"].map(content_map)
    pct = interactions["watched_pct"]
    item_lookup = items.set_index("item_id")

    def duration_check(item_id: int) -> dict:
        rows = interactions.loc[interactions["item_id"] == item_id]
        usable = rows.loc[rows["watched_pct"] >= 20]
        implied_min = usable["total_dur"] / (usable["watched_pct"] / 100.0) / 60.0
        high = rows.loc[rows["watched_pct"] >= 95, "total_dur"] / 60.0
        title = item_lookup.at[item_id, "title"] if item_id in item_lookup.index else None
        return {
            "item_id": item_id,
            "title": None
            if title is None or (isinstance(title, float) and math.isnan(title))
            else str(title),
            "content_type": (
                None
                if item_id not in item_lookup.index
                else str(item_lookup.at[item_id, "content_type"])
            ),
            "n": int(len(rows)),
            "median_minutes_at_pct_ge_95": None if high.empty else float(high.median()),
            "median_implied_full_minutes": None
            if implied_min.empty
            else float(implied_min.median()),
        }

    top_film = (
        interactions.loc[content.eq("film"), "item_id"].value_counts().index[0]
        if content.eq("film").any()
        else None
    )
    top_series = (
        interactions.loc[content.eq("series"), "item_id"].value_counts().index[0]
        if content.eq("series").any()
        else None
    )
    checks = [duration_check(10711)]
    for item_id in (top_film, top_series):
        if item_id is not None and int(item_id) != 10711:
            checks.append(duration_check(int(item_id)))

    blank_share = {
        column: float(is_blank(items[column]).mean())
        for column in (
            "title",
            "title_orig",
            "description",
            "keywords",
            "genres",
            "countries",
            "studios",
            "directors",
            "actors",
            "for_kids",
        )
    }
    blank_share["release_year"] = float(items["release_year"].isna().mean())
    blank_share["age_rating"] = float(items["age_rating"].isna().mean())
    description = items["description"].fillna("").astype(str).str.strip()
    description_len = description.str.len()
    nonempty_len = description_len.loc[~is_blank(items["description"])]

    genre_lists = items["genres"].map(split_genres)
    genre_counts = pd.Series([genre for genres in genre_lists for genre in genres]).value_counts()
    n_genres = genre_lists.map(len)

    profile_users = users.drop_duplicates("user_id").set_index("user_id").reindex(list(inter_users))
    # У пользователя без строки анкеты все поля пустые.
    present = profile_users.index.isin(user_ids)
    user_missing = {
        "no_profile_row": float((~present).mean()),
        "age": float(profile_users["age"].isna().mean()),
        "income": float(profile_users["income"].isna().mean()),
        "sex": float(profile_users["sex"].isna().mean()),
        "kids_flg": float(profile_users["kids_flg"].isna().mean()),
    }
    file_missing = {
        column: float(users[column].isna().mean())
        for column in ("age", "income", "sex", "kids_flg")
    }

    filled = profile_users.loc[
        profile_users["age"].notna() & profile_users["kids_flg"].notna()
    ].copy()
    filled["kids_flg"] = filled["kids_flg"].astype(int)
    age_kids = filled.groupby(["age", "kids_flg"], dropna=False).size()
    fine = filled.loc[filled["sex"].notna() & filled["income"].notna()]
    fine_groups = fine.groupby(["age", "kids_flg", "sex", "income"], dropna=False).size()

    def group_report(sizes: pd.Series) -> dict:
        return {
            "n_groups": int(sizes.size),
            "min": int(sizes.min()) if len(sizes) else None,
            "p10": float(sizes.quantile(0.1)) if len(sizes) else None,
            "median": float(sizes.median()) if len(sizes) else None,
            "share_users_in_groups_under_500": (
                float(sizes.loc[sizes < 500].sum() / sizes.sum()) if len(sizes) else None
            ),
        }

    return {
        "hashes": {},
        "n_interactions": int(len(interactions)),
        "n_interaction_users": len(inter_users),
        "n_interaction_items": len(inter_items),
        "n_user_rows": int(len(users)),
        "n_unique_user_ids": int(users["user_id"].nunique()),
        "n_item_rows": int(len(items)),
        "n_unique_item_ids": int(items["item_id"].nunique()),
        "duplicate_user_item_pairs": int(interactions.duplicated(["user_id", "item_id"]).sum()),
        "duplicate_user_ids": int(users["user_id"].duplicated().sum()),
        "duplicate_item_ids": int(items["item_id"].duplicated().sum()),
        "users_without_profile": len(inter_users - user_ids),
        "profiles_without_interactions": len(user_ids - inter_users),
        "items_without_interactions": len(item_ids - inter_items),
        "interactions_without_item_card": len(inter_items - item_ids),
        "nulls_interactions": {
            column: int(interactions[column].isna().sum()) for column in interactions.columns
        },
        "watched_pct_lt_0": int((pct < 0).sum()),
        "watched_pct_gt_100": int((pct > 100).sum()),
        "watched_pct_eq_0": int((pct == 0).sum()),
        "watched_pct_eq_100": int((pct == 100).sum()),
        "total_dur_le_0": int((interactions["total_dur"] <= 0).sum()),
        "total_dur_lt_60s": int((interactions["total_dur"] < 60).sum()),
        "date_min": daily.index.min(),
        "date_max": daily.index.max(),
        "n_days": int(daily.size),
        "missing_days": missing_days,
        "last_day_rows": last_day,
        "median_other_days": float(previous.median()),
        "weekday_of_last_day": daily.index.max().day_name(),
        "history_length": {
            "mean": float(history.mean()),
            "median": float(history.median()),
            "p90": float(history.quantile(0.9)),
            "p99": float(history.quantile(0.99)),
            "max": int(history.max()),
            "share_1": float((history == 1).mean()),
            "share_2_3": float(history.between(2, 3).mean()),
            "share_4_10": float(history.between(4, 10).mean()),
            "share_gt_10": float((history > 10).mean()),
        },
        "duration_minutes": _duration_summary(interactions, content, pct),
        "content_type_rows": value_counts(items["content_type"]),
        "content_type_interactions": {
            "film": int(content.eq("film").sum()),
            "series": int(content.eq("series").sum()),
            "missing": int(content.isna().sum()),
        },
        "age_rating": value_counts(items["age_rating"]),
        "for_kids": value_counts(items["for_kids"]),
        "user_field_counts": {
            "age": value_counts(users["age"]),
            "income": value_counts(users["income"]),
            "sex": value_counts(users["sex"]),
            "kids_flg": value_counts(users["kids_flg"]),
        },
        "user_missing_among_interaction_users": user_missing,
        "user_missing_inside_users_csv": file_missing,
        "demo_group_age_kids": group_report(age_kids),
        "demo_group_age_kids_sex_income": group_report(fine_groups),
        "item_blank_share": blank_share,
        "description_chars": {
            "median_nonempty": float(nonempty_len.median()),
            "p10_nonempty": float(nonempty_len.quantile(0.1)),
            "share_nonempty_under_50_chars": float((nonempty_len < 50).mean()),
        },
        "genres_per_item": {
            "median": float(n_genres.median()),
            "mean": float(n_genres.mean()),
            "p90": float(n_genres.quantile(0.9)),
            "max": int(n_genres.max()),
            "share_0": float((n_genres == 0).mean()),
        },
        "top_genres": {str(key): int(value) for key, value in genre_counts.head(25).items()},
        "duration_checks": checks,
        "daily_counts": {day.strftime("%Y-%m-%d"): int(count) for day, count in daily.items()},
        "_genre_lists": genre_lists,
        "_content": content,
        "_history": history,
        "_genre_names": genre_counts.index.astype(str).tolist(),
    }


def plot_overview(interactions: pd.DataFrame, facts: dict) -> None:
    log("графики состава")
    daily = pd.Series(facts["daily_counts"])
    daily.index = pd.to_datetime(daily.index)
    pct = interactions["watched_pct"].to_numpy()
    dur_min = interactions["total_dur"].to_numpy() / 60.0
    content = facts["_content"]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(daily.index, daily.values, color=PRIMARY, linewidth=1.0, label="Все взаимодействия")
    positive = (interactions["watched_pct"] >= 50) & (dur_min >= 10)
    daily_pos = interactions.loc[positive].groupby("last_watch_dt").size()
    ax.plot(
        daily_pos.index,
        daily_pos.values,
        color=SECOND,
        linewidth=1.0,
        label="Позитив: процент ≥ 50 и ≥ 10 минут",
    )
    end = daily.index.max()
    cuts = {
        "ранкер": end - pd.Timedelta(days=41),
        "валидация": end - pd.Timedelta(days=27),
        "тест": end - pd.Timedelta(days=13),
    }
    for label, moment in cuts.items():
        ax.axvline(moment, color=NEUTRAL, linewidth=0.8, linestyle="--")
        ax.text(
            moment, ax.get_ylim()[1] * 0.98, label, rotation=90, va="top", ha="right", fontsize=8
        )
    ax.set_title("Взаимодействия по дням")
    ax.set_ylabel("Строк в день")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False)
    style_ax(ax)
    fig.autofmt_xdate()
    save_fig(fig, "daily_volume.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 100, 51)
    ax.hist(np.clip(pct, 0, 100), bins=bins, color=PRIMARY, edgecolor="white", linewidth=0.3)
    ax.axvline(50, color=SECOND, linestyle="--", label="Порог 50%")
    ax.set_title("Распределение watched_pct")
    ax.set_xlabel("watched_pct")
    ax.set_ylabel("Строк")
    ax.legend(frameon=False)
    style_ax(ax)
    save_fig(fig, "watched_pct.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.logspace(np.log10(1 / 60), np.log10(max(float(np.quantile(dur_min, 0.995)), 30)), 50)
    for label, mask, color in (
        ("Фильмы", content.eq("film").to_numpy(), PRIMARY),
        ("Сериалы", content.eq("series").to_numpy(), SECOND),
    ):
        ax.hist(
            np.clip(dur_min[mask], 1 / 60, None),
            bins=bins,
            histtype="step",
            linewidth=1.6,
            label=label,
            color=color,
        )
    for minute, text in ((5, "5 мин"), (10, "10 мин"), (15, "15 мин")):
        ax.axvline(minute, color=NEUTRAL, linewidth=0.7, linestyle="--")
        ax.text(
            minute, ax.get_ylim()[1] * 0.9, text, rotation=90, va="top", fontsize=8, color=NEUTRAL
        )
    ax.set_xscale("log")
    ax.set_title("Распределение длительности")
    ax.set_xlabel("total_dur, минуты")
    ax.set_ylabel("Строк")
    ax.legend(frameon=False)
    style_ax(ax)
    save_fig(fig, "duration_minutes.png")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    plot_mask = (pct >= 0) & (pct <= 100)
    hexbin = ax.hexbin(
        np.clip(dur_min[plot_mask], 0.05, None),
        pct[plot_mask],
        xscale="log",
        gridsize=45,
        cmap="Blues",
        bins="log",
        mincnt=1,
        extent=(np.log10(0.05), np.log10(3000), 0, 100),
    )
    ax.set_xlim(0.05, 3000)
    ax.axhline(50, color=SECOND, linestyle="--", linewidth=1)
    ax.axvline(10, color=SECOND, linestyle="--", linewidth=1)
    ax.set_title("watched_pct и длительность")
    ax.set_xlabel("Длительность, минуты")
    ax.set_ylabel("Процент досмотра")
    fig.colorbar(hexbin, ax=ax, label="Строк, log")
    save_fig(fig, "watch_joint.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    history = facts["_history"]
    clipped = np.clip(history.to_numpy(), 1, 30)
    ax.hist(clipped, bins=np.arange(0.5, 31.5, 1), color=PRIMARY, edgecolor="white", linewidth=0.3)
    ax.set_title("Длина истории")
    ax.set_xlabel("Тайтлов в истории; длиннее 30 собраны справа")
    ax.set_ylabel("Пользователей")
    style_ax(ax)
    save_fig(fig, "history_length.png")


def plot_quality(items: pd.DataFrame, facts: dict) -> None:
    user_missing = facts["user_missing_among_interaction_users"]
    item_blank = facts["item_blank_share"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    user_labels = ["нет анкеты", "возраст", "доход", "пол", "kids_flg"]
    user_keys = ["no_profile_row", "age", "income", "sex", "kids_flg"]
    user_values = [user_missing[key] * 100 for key in user_keys]
    axes[0].barh(user_labels[::-1], user_values[::-1], color=PRIMARY)
    axes[0].set_xlabel("Доля пользователей взаимодействий, %")
    axes[0].set_title("Пустая демография")
    item_order = [
        "for_kids",
        "studios",
        "title_orig",
        "actors",
        "directors",
        "keywords",
        "countries",
        "release_year",
        "age_rating",
        "description",
        "genres",
        "title",
    ]
    item_labels = {
        "for_kids": "for_kids",
        "studios": "студии",
        "title_orig": "оригинальное название",
        "actors": "актеры",
        "directors": "режиссеры",
        "keywords": "ключевые слова",
        "countries": "страны",
        "release_year": "год",
        "age_rating": "возрастной рейтинг",
        "description": "описание",
        "genres": "жанры",
        "title": "название",
    }
    item_values = [item_blank[key] * 100 for key in item_order]
    axes[1].barh(
        [item_labels[key] for key in item_order][::-1],
        item_values[::-1],
        color=SECOND,
    )
    axes[1].set_xlabel("Доля тайтлов каталога, %")
    axes[1].set_title("Пустые поля карточки")
    for ax in axes:
        style_ax(ax)
        ax.set_xlim(0, 100)
    fig.tight_layout()
    save_fig(fig, "missingness.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    genre_counts = pd.Series(facts["top_genres"]).sort_values()
    axes[0].barh(
        genre_counts.index, genre_counts.values / facts["n_item_rows"] * 100, color=PRIMARY
    )
    axes[0].set_xlabel("Доля каталога, %")
    axes[0].set_title("Частые метки в поле жанров")
    n_genres = facts["_genre_lists"].map(len)
    axes[1].hist(
        np.clip(n_genres.to_numpy(), 0, 12),
        bins=np.arange(-0.5, 13.5, 1),
        color=PRIMARY,
        edgecolor="white",
    )
    axes[1].set_xlabel("Меток у тайтла, 12 и больше собраны справа")
    axes[1].set_ylabel("Тайтлов")
    axes[1].set_title("Сколько меток у одной карточки")
    for ax in axes:
        style_ax(ax)
    fig.tight_layout()
    save_fig(fig, "genres.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    content_counts = pd.Series(facts["content_type_rows"])
    axes[0].bar(content_counts.index.astype(str), content_counts.values, color=PRIMARY)
    axes[0].set_ylabel("Тайтлов")
    axes[0].set_title("Фильмы и сериалы в каталоге")
    age = items["age_rating"].fillna(-1).astype(int)
    age_counts = age.value_counts().sort_index()
    age_labels = ["пусто" if value < 0 else str(value) for value in age_counts.index]
    axes[1].bar(age_labels, age_counts.values, color=PRIMARY)
    axes[1].set_xlabel("age_rating")
    axes[1].set_ylabel("Тайтлов")
    axes[1].set_title("age_rating")
    for ax in axes:
        style_ax(ax)
    fig.tight_layout()
    save_fig(fig, "catalog.png")


def threshold_grid(interactions: pd.DataFrame) -> dict:
    pct = interactions["watched_pct"].to_numpy()
    dur_min = interactions["total_dur"].to_numpy() / 60.0
    grid = []
    for min_pct in PCT_LEVELS:
        row = []
        for min_dur in DUR_LEVELS:
            share = float(np.mean((pct >= min_pct) & (dur_min >= min_dur)))
            row.append(share)
        grid.append(row)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    image = np.asarray(grid)
    heatmap = ax.imshow(image, cmap="Blues", vmin=0, vmax=max(0.01, float(image.max())))
    ax.set_xticks(range(len(DUR_LEVELS)), [str(value) for value in DUR_LEVELS])
    ax.set_yticks(range(len(PCT_LEVELS)), [str(value) for value in PCT_LEVELS])
    ax.set_xlabel("Минимальная длительность, минуты")
    ax.set_ylabel("Минимальный watched_pct")
    ax.set_title("Доля взаимодействий выше обоих порогов")
    for row_index, row in enumerate(grid):
        for col_index, share in enumerate(row):
            color = "white" if share > image.max() * 0.65 else "black"
            ax.text(
                col_index,
                row_index,
                f"{share * 100:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.colorbar(heatmap, ax=ax, fraction=0.046, label="Доля строк")
    save_fig(fig, "threshold_heatmap.png")
    return {
        "pct_levels": list(PCT_LEVELS),
        "dur_levels": list(DUR_LEVELS),
        "share": grid,
    }


def window_bounds(end: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    test_start = end - pd.Timedelta(days=13)
    val_start = end - pd.Timedelta(days=27)
    rank_start = end - pd.Timedelta(days=41)
    horizon_end = end + pd.Timedelta(days=1)
    return {
        "ranker": (rank_start, val_start),
        "val": (val_start, test_start),
        "test": (test_start, horizon_end),
    }


def prepare_seen(
    interactions: pd.DataFrame,
    before: pd.Series,
    eval_ids: list[int],
) -> dict[int, set[int]]:
    eval_set = set(eval_ids)
    rows = interactions.loc[
        before & (interactions["watched_pct"] > 0) & interactions["user_id"].isin(eval_set),
        ["user_id", "item_id"],
    ]
    seen: dict[int, set[int]] = {}
    for user_id, item_id in zip(
        rows["user_id"].to_numpy(), rows["item_id"].to_numpy(), strict=True
    ):
        seen.setdefault(int(user_id), set()).add(int(item_id))
    return seen


def evaluate_candidate(
    interactions: pd.DataFrame,
    positive: pd.Series,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    demo: pd.DataFrame,
    demo_map: dict[int, tuple[str, int]],
    catalog_size: int,
    with_demo: bool,
    with_tail: bool,
) -> dict:
    dates = interactions["last_watch_dt"]
    before = dates < start
    inside = (dates >= start) & (dates < stop)
    pos_before = before & positive
    pos_inside = inside & positive
    counts = interactions.loc[pos_before, "item_id"].value_counts()
    ordered = ranking_frame(counts)
    ranking = [int(item_id) for item_id in ordered["item_id"].head(RANKING_LIMIT).tolist()]
    full_ranking = [int(item_id) for item_id in ordered["item_id"].tolist()]
    relevant = build_rel(interactions.loc[pos_inside, ["user_id", "item_id"]])
    train_users = set(interactions.loc[pos_before, "user_id"].unique().tolist())
    before_users = set(interactions.loc[before, "user_id"].unique().tolist())
    eval_ids = [user_id for user_id in relevant if user_id in train_users]
    seen = prepare_seen(interactions, before, eval_ids)
    train_pos_per_user = interactions.loc[pos_before].groupby("user_id").size()
    eval_history = train_pos_per_user.reindex(eval_ids).fillna(0).to_numpy()
    top100 = set(full_ranking[:100])
    tail_items = (
        None
        if not with_tail
        else set(full_ranking[100:])
        | (set(int(item_id) for item_id in relevant_items(relevant)) - top100)
    )
    # Items absent from the train ranking are also outside the top 100.
    if tail_items is not None:
        tail_items |= set(relevant_items(relevant)) - set(full_ranking)
    global_metrics = metric_bundle(
        eval_ids,
        relevant,
        seen,
        ranking,
        tail_items=tail_items,
    )
    demo_metrics = None
    if with_demo:
        groupings = group_rankings_for(interactions, pos_before, demo)
        demo_metrics = metric_bundle(
            eval_ids,
            relevant,
            seen,
            ranking,
            group_rankings=groupings,
            demo_map=demo_map,
            tail_items=tail_items,
        )
    head_share = {}
    inside_items = interactions.loc[pos_inside, "item_id"].to_numpy()
    for width in HEAD_KS:
        head_share[str(width)] = (
            float(np.isin(inside_items, np.asarray(full_ranking[:width])).mean())
            if len(inside_items)
            else None
        )
    cold_item_rows = (
        int((~np.isin(inside_items, np.asarray(full_ranking))).sum()) if len(inside_items) else 0
    )
    cold_users = [user_id for user_id in relevant if user_id not in before_users]
    no_positive_users = [
        user_id for user_id in relevant if user_id in before_users and user_id not in train_users
    ]
    positive_rows = int(pos_inside.sum())
    cold_rows = sum(len(relevant[user_id]) for user_id in cold_users)
    return {
        "n_positive_rows_before": int(pos_before.sum()),
        "n_positive_rows_inside": positive_rows,
        "n_users_with_positive_inside": len(relevant),
        "n_eval_users": len(eval_ids),
        "n_cold_users": len(cold_users),
        "n_users_with_history_but_no_positive": len(no_positive_users),
        "share_positive_rows_from_cold_users": (cold_rows / positive_rows)
        if positive_rows
        else None,
        "share_positive_rows_on_cold_items": (cold_item_rows / positive_rows)
        if positive_rows
        else None,
        "n_train_positive_items": int(counts.size),
        "median_train_positive_count_among_items_with_one": (
            float(np.median(counts.to_numpy())) if len(counts) else None
        ),
        "eval_train_positives": {
            "median": float(np.median(eval_history)) if len(eval_history) else None,
            "share_1": float(np.mean(eval_history == 1)) if len(eval_history) else None,
            "share_2_3": float(np.mean((eval_history >= 2) & (eval_history <= 3)))
            if len(eval_history)
            else None,
            "share_4_10": float(np.mean((eval_history >= 4) & (eval_history <= 10)))
            if len(eval_history)
            else None,
            "share_gt_10": float(np.mean(eval_history > 10)) if len(eval_history) else None,
        },
        "head_share_of_positive_rows": head_share,
        "global": attach_coverage(global_metrics, catalog_size),
        "demographic": None
        if demo_metrics is None
        else attach_coverage(demo_metrics, catalog_size),
        "_top_table_source": ordered.head(10),
        "_counts": counts,
    }


def relevant_items(relevant: dict[int, set[int]]) -> list[int]:
    found: set[int] = set()
    for items in relevant.values():
        found.update(items)
    return list(found)


def attach_coverage(metrics: dict, catalog_size: int) -> dict:
    metrics = dict(metrics)
    metrics["coverage@10"] = metrics["coverage_items"] / catalog_size if catalog_size else None
    return metrics


def top_table(ordered: pd.DataFrame, items: pd.DataFrame) -> list[dict]:
    cards = items.set_index("item_id")
    rows = []
    for _, row in ordered.iterrows():
        item_id = int(row["item_id"])
        card = cards.loc[item_id] if item_id in cards.index else None
        genres = [] if card is None else split_genres(card["genres"])
        rows.append(
            {
                "item_id": item_id,
                "cnt": int(row["cnt"]),
                "title": None if card is None else str(card["title"]),
                "content_type": None if card is None else str(card["content_type"]),
                "age_rating": None
                if card is None or pd.isna(card["age_rating"])
                else float(card["age_rating"]),
                "genres": genres,
            }
        )
    return rows


def genre_cap_report(table: list[dict]) -> dict:
    tag_counts: dict[str, int] = defaultdict(int)
    for row in table:
        for genre in set(row["genres"]):
            tag_counts[genre] += 1
    ordered = sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "n_titles": len(table),
        "tags_in_at_least_6_of_top10": [tag for tag, count in ordered if count >= 6],
        "tag_counts": {tag: count for tag, count in ordered},
    }


def plot_windows(candidate_windows: dict) -> None:
    order = ["ranker", "val", "test"]
    labels = ["окно ранкера", "валидация", "тест"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(order))
    width = 0.36
    positives = [candidate_windows[name]["n_positive_rows_inside"] for name in order]
    users = [candidate_windows[name]["n_eval_users"] for name in order]
    ax.bar(x - width / 2, positives, width, color=PRIMARY, label="Позитивов в окне")
    ax.bar(
        x + width / 2,
        users,
        width,
        color=SECOND,
        label="Пользователей с позитивом до окна и в окне",
    )
    ax.set_xticks(x, labels)
    ax.set_title("Позитивы и пользователи по окнам")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    style_ax(ax)
    fig.tight_layout()
    save_fig(fig, "windows.png")


def plot_head(head_share: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ks = [int(key) for key in head_share]
    shares = [head_share[str(key)] * 100 for key in ks]
    ax.plot(ks, shares, marker="o", color=PRIMARY)
    ax.set_xscale("log")
    ax.set_xlabel("Сколько самых частых тайтлов до отсечки")
    ax.set_ylabel("Доля позитивов тестового окна, %")
    ax.set_title("Доля позитивов теста в голове популярности")
    style_ax(ax)
    save_fig(fig, "popularity_head.png")


def cooccurrence(
    user_lists: list[list[int]],
) -> tuple[dict[tuple[int, int], int], dict[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    popularity: dict[int, int] = defaultdict(int)
    capped = 0
    for items in user_lists:
        unique_items = list(dict.fromkeys(items))
        if len(unique_items) > LIFT_HISTORY_CAP:
            unique_items = unique_items[-LIFT_HISTORY_CAP:]
            capped += 1
        if len(unique_items) < 2:
            for item_id in unique_items:
                popularity[item_id] += 1
            continue
        unique_items = sorted(unique_items)
        for item_id in unique_items:
            popularity[item_id] += 1
        for left_index, left in enumerate(unique_items):
            for right in unique_items[left_index + 1 :]:
                key = (left, right)
                counts[key] = counts.get(key, 0) + 1
    return counts, popularity, capped


def user_positive_lists(
    interactions: pd.DataFrame,
    positive: pd.Series,
    fold: int | None = None,
) -> list[list[int]]:
    rows = interactions.loc[positive, ["user_id", "item_id", "last_watch_dt"]]
    if fold is not None:
        rows = rows.loc[rows["user_id"] % 2 == fold]
    rows = rows.sort_values(["user_id", "last_watch_dt", "item_id"])
    lists = []
    current_user = None
    bucket: list[int] = []
    for user_id, item_id in zip(
        rows["user_id"].to_numpy(), rows["item_id"].to_numpy(), strict=True
    ):
        user_id = int(user_id)
        if current_user is None:
            current_user = user_id
        if user_id != current_user:
            lists.append(bucket)
            bucket = []
            current_user = user_id
        bucket.append(int(item_id))
    if bucket:
        lists.append(bucket)
    return lists


def co_histogram(counts: dict[tuple[int, int], int]) -> dict:
    if not counts:
        return {"n_pairs": 0}
    values = np.fromiter(counts.values(), dtype=np.int32, count=len(counts))
    total_mass = float(values.sum())
    thresholds = [1, 2, 3, 5, 10, 20, 40, 50]
    kept = []
    for threshold in thresholds:
        mask = values >= threshold
        kept.append(
            {
                "min_count": threshold,
                "share_pairs": float(mask.mean()),
                "share_co_watch_mass": float(values[mask].sum() / total_mass)
                if total_mass
                else None,
                "n_pairs": int(mask.sum()),
            }
        )
    return {
        "n_pairs": int(values.size),
        "share_pairs_eq_1": float(np.mean(values == 1)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": int(values.max()),
        "kept": kept,
    }


def neighbor_ids(
    item_id: int,
    adjacency: dict[int, list[tuple[int, int]]],
    popularity: dict[int, int],
    min_count: int,
    k: int = 20,
) -> list[int]:
    own = popularity.get(item_id, 0)
    if own <= 0:
        return []
    scored: list[tuple[float, int]] = []
    for other, count in adjacency.get(item_id, []):
        if count < min_count:
            continue
        other_pop = popularity.get(other, 0)
        if other_pop <= 0:
            continue
        scored.append((count / (own * other_pop), other))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [other for _, other in scored[:k]]


def adjacency_from(counts: dict[tuple[int, int], int]) -> dict[int, list[tuple[int, int]]]:
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (left, right), count in counts.items():
        adjacency[left].append((right, count))
        adjacency[right].append((left, count))
    return adjacency


def lift_stability(interactions: pd.DataFrame, positive: pd.Series) -> dict:
    log("считаю совместные досмотры")
    full_lists = user_positive_lists(interactions, positive)
    full_counts, _full_pop, capped = cooccurrence(full_lists)
    histogram = co_histogram(full_counts)
    del full_counts
    adjacencies = []
    pops = []
    for fold in (0, 1):
        log(f"половина пользователей {fold}")
        lists = user_positive_lists(interactions, positive, fold=fold)
        counts, popularity, _ = cooccurrence(lists)
        adjacencies.append(adjacency_from(counts))
        pops.append(popularity)
        del counts
    eligible = [
        item_id
        for item_id, pop in pops[0].items()
        if pop >= LIFT_MIN_POP and pops[1].get(item_id, 0) >= LIFT_MIN_POP
    ]
    rng = np.random.default_rng(0)
    if len(eligible) > LIFT_ITEM_SAMPLE:
        eligible = [
            int(item_id) for item_id in rng.choice(eligible, size=LIFT_ITEM_SAMPLE, replace=False)
        ]
    curves = []
    for min_count in LIFT_MIN_COUNTS:
        scores = []
        neighbor_cos = []
        used = 0
        for item_id in eligible:
            left = neighbor_ids(item_id, adjacencies[0], pops[0], min_count)
            right = neighbor_ids(item_id, adjacencies[1], pops[1], min_count)
            if len(left) < 5 or len(right) < 5:
                continue
            union = set(left) | set(right)
            scores.append(len(set(left) & set(right)) / len(union))
            used += 1
            neighbor_cos.extend(
                count
                for other, count in adjacencies[0].get(item_id, [])
                if other in set(left) and count >= min_count
            )
        curves.append(
            {
                "min_count_on_half": min_count,
                "min_count_full_equivalent": min_count * 2,
                "n_items": used,
                "mean_jaccard_top20": float(np.mean(scores)) if scores else None,
                "median_jaccard_top20": float(np.median(scores)) if scores else None,
            }
        )
        del neighbor_cos
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    kept = histogram["kept"]
    axes[0].plot(
        [row["min_count"] for row in kept],
        [row["share_pairs"] * 100 for row in kept],
        marker="o",
        color=PRIMARY,
        label="Доля пар",
    )
    axes[0].plot(
        [row["min_count"] for row in kept],
        [row["share_co_watch_mass"] * 100 for row in kept],
        marker="o",
        color=SECOND,
        label="Доля массы совместных досмотров",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Минимум совместных позитивов на полной предтестовой матрице")
    axes[0].set_ylabel("%")
    axes[0].set_title("Пары по числу совместных позитивов")
    axes[0].legend(frameon=False)
    axes[1].plot(
        [row["min_count_full_equivalent"] for row in curves],
        [0 if row["mean_jaccard_top20"] is None else row["mean_jaccard_top20"] for row in curves],
        marker="o",
        color=PRIMARY,
    )
    axes[1].set_xlabel("Эквивалент порога на полной матрице")
    axes[1].set_ylabel("Средний Jaccard топ-20 соседей")
    axes[1].set_title("Соседи по lift на двух половинах пользователей")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        style_ax(ax)
    fig.tight_layout()
    save_fig(fig, "lift_stability.png")
    return {
        "history_cap": LIFT_HISTORY_CAP,
        "users_capped": capped,
        "n_user_lists": len(full_lists),
        "histogram": histogram,
        "half_split": {
            "min_pop_each_half": LIFT_MIN_POP,
            "n_eligible_items_before_sample": len(
                [
                    item_id
                    for item_id, pop in pops[0].items()
                    if pop >= LIFT_MIN_POP and pops[1].get(item_id, 0) >= LIFT_MIN_POP
                ]
            ),
            "sample_items": len(eligible),
            "curves": curves,
        },
    }


def contest_like(
    interactions: pd.DataFrame,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    catalog_size: int,
) -> dict:
    dates = interactions["last_watch_dt"]
    before = dates < start
    inside = (dates >= start) & (dates < stop)
    counts = interactions.loc[before, "item_id"].value_counts()
    ranking = ranking_ids(counts, limit=RANKING_LIMIT)
    relevant = build_rel(interactions.loc[inside, ["user_id", "item_id"]])
    before_users = set(interactions.loc[before, "user_id"].unique().tolist())
    eval_ids = [user_id for user_id in relevant if user_id in before_users]
    seen = prepare_seen(interactions, before, eval_ids)
    metrics = metric_bundle(eval_ids, relevant, seen, ranking, tail_items=None)
    return attach_coverage(metrics, catalog_size)


def strip_private(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def main() -> None:
    started = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "axes.unicode_minus": False,
        }
    )
    hashes = require_inputs()
    interactions, users, items = load_frames()
    facts = profile(interactions, users, items)
    facts["hashes"] = hashes or {
        "sha256": ZIP_SHA256,
        "md5": ZIP_MD5,
        "zip_present": ZIP_PATH.exists(),
    }
    plot_overview(interactions, facts)
    plot_quality(items, facts)
    grid = threshold_grid(interactions)
    end = interactions["last_watch_dt"].max()
    windows = window_bounds(end)
    demo = users.loc[
        users["age"].notna() & users["kids_flg"].notna(), ["user_id", "age", "kids_flg"]
    ].copy()
    demo["kids_flg"] = demo["kids_flg"].astype(int)
    demo["age"] = demo["age"].astype(str)
    demo_map = {
        int(user_id): (str(age), int(kids))
        for user_id, age, kids in zip(demo["user_id"], demo["age"], demo["kids_flg"], strict=True)
    }
    pct = interactions["watched_pct"]
    dur_min = interactions["total_dur"] / 60.0
    content = facts["_content"]
    catalog_size = int(items["item_id"].nunique())
    candidate_reports = {}
    selected_windows = None
    selected_top = None
    for candidate in CANDIDATES:
        log(f"бейзлайны {candidate['name']}")
        positive = (pct >= candidate["pct"]) & (dur_min >= candidate["dur_min"])
        by_type = {}
        for label in ("film", "series"):
            mask = content.eq(label)
            by_type[label] = {
                "rows": int(mask.sum()),
                "positive_rows": int((positive & mask).sum()),
                "positive_share": float((positive & mask).sum() / mask.sum())
                if mask.any()
                else None,
            }
        report = {
            "pct": candidate["pct"],
            "dur_min": candidate["dur_min"],
            "n_positive_rows": int(positive.sum()),
            "share_of_interactions": float(positive.mean()),
            "by_content_type": by_type,
            "windows": {},
        }
        for window_name, (start, stop) in windows.items():
            log(f"  окно {window_name}")
            measured = evaluate_candidate(
                interactions,
                positive,
                start,
                stop,
                demo,
                demo_map,
                catalog_size,
                with_demo=True,
                with_tail=window_name == "test",
            )
            if window_name == "test" and candidate["name"] == LIFT_CANDIDATE:
                selected_top = top_table(measured.pop("_top_table_source"), items)
                measured.pop("_counts", None)
            else:
                measured.pop("_top_table_source", None)
                measured.pop("_counts", None)
            report["windows"][window_name] = strip_private(measured)
        candidate_reports[candidate["name"]] = report
        if candidate["name"] == LIFT_CANDIDATE:
            selected_windows = report["windows"]

    if selected_windows is None or selected_top is None:
        raise SystemExit("Не посчитан выбранный кандидат для графиков хвоста")
    plot_windows(selected_windows)
    plot_head(selected_windows["test"]["head_share_of_positive_rows"])
    cap = genre_cap_report(selected_top)

    test_start, test_stop = windows["test"]
    week_start = end - pd.Timedelta(days=6)
    log("MAP по любому просмотру")
    contest = {
        "last_14_days": contest_like(interactions, test_start, test_stop, catalog_size),
        "last_7_days": contest_like(interactions, week_start, test_stop, catalog_size),
        "note": (
            "Релевантен любой просмотр окна, не только позитив. "
            "Популярность — число любых взаимодействий до отсечки. "
            "Это не цифра закрытой недели конкурса."
        ),
    }
    lift_positive = (pct >= 50) & (dur_min >= 10) & (interactions["last_watch_dt"] < test_start)
    lift = lift_stability(interactions, lift_positive)

    # Median of train positive counts over the whole catalog, for the selected test split.
    selected_counts = interactions.loc[
        (pct >= 50) & (dur_min >= 10) & (interactions["last_watch_dt"] < test_start),
        "item_id",
    ].value_counts()
    catalog_counts = selected_counts.reindex(items["item_id"].unique(), fill_value=0)
    median_block = {
        "median_over_catalog_including_zero": float(np.median(catalog_counts.to_numpy())),
        "share_catalog_with_zero_train_positives": float(np.mean(catalog_counts.to_numpy() == 0)),
        "median_over_items_with_at_least_one": float(np.median(selected_counts.to_numpy())),
        "p90_over_items_with_at_least_one": float(np.quantile(selected_counts.to_numpy(), 0.9)),
    }

    facts_out = {key: value for key, value in facts.items() if not key.startswith("_")}
    facts_out["paper_targets"] = {
        "interactions": PAPER_INTERACTIONS,
        "users": PAPER_USERS,
        "items_in_interactions": PAPER_ITEMS,
        "interactions_match": facts["n_interactions"] == PAPER_INTERACTIONS,
        "users_match": facts["n_interaction_users"] == PAPER_USERS,
        "interaction_items_match": facts["n_interaction_items"] == PAPER_ITEMS,
    }
    summary = {
        "zip_sha256": ZIP_SHA256,
        "zip_md5": ZIP_MD5,
        "positive_proposal": {
            "name": LIFT_CANDIDATE,
            "watched_pct_min": 50,
            "total_dur_min_minutes": 10,
            "why": (
                "Пики watched_pct в 0 и в 100, между ними порог 50. "
                "10 минут убирают короткие строки с высоким процентом."
            ),
        },
        "windows": {
            name: {"start": start, "end_inclusive": stop - pd.Timedelta(days=1)}
            for name, (start, stop) in windows.items()
        },
        "profile": facts_out,
        "threshold_grid": grid,
        "candidates": candidate_reports,
        "selected_test_top10": selected_top,
        "genre_cap_on_selected_top10": cap,
        "popularity_median": median_block,
        "contest_like_any_watch": contest,
        "lift": lift,
        "runtime_sec": round(time.time() - started, 1),
    }
    SUMMARY_PATH.write_text(
        json.dumps(to_builtin(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"готово за {summary['runtime_sec']} с, сводка {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
