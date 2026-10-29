# Как вносить изменения

Изменения через pull request в `main`.

## Окружение

Нужен Python 3.11+: `pre-commit` 4.7 на более старых версиях не ставится.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```

Если системный Python старше, окружение можно собрать через [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

## Линтер

[Ruff](https://docs.astral.sh/ruff/) и проверяет, и форматирует. Настройки в `pyproject.toml`. В `.pre-commit-config.yaml` стоят `ruff check --fix` и `ruff format`.

```bash
make lint
make format
```

Если хук что-то поправил, коммит не пройдет. Файлы нужно добавить заново и закоммитить еще раз.

## Данные

Архив KION и распакованные csv не коммитим: `data/` и `*.zip` в `.gitignore`. Ссылка на выгрузку — в [README](README.md) и в [дизайн-документе](docs/ml_system_design_doc.md).

## Дизайн-документ

Если код расходится с `docs/ml_system_design_doc.md`, сначала правим документ. Размеры выборок, даты отсечек и полученные метрики дописываем туда после EDA.
