# Managing the Python environment with uv

This repository uses [uv](https://docs.astral.sh/uv/) for Python version,
virtualenv, and locked dependency management.

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

## Sync

```bash
# Recommended: all extras + dev tools
bash scripts/setup_uv.sh all

# Or per framework
bash scripts/setup_uv.sh trl
bash scripts/setup_uv.sh openrlhf
bash scripts/setup_uv.sh verl
```

Equivalent commands:

```bash
uv python install   # reads .python-version (default 3.12)
uv sync             # base deps
uv sync --extra trl
uv sync --extra openrlhf
uv sync --extra all --group dev
```

## Daily use

```bash
# Run without activating the venv
uv run python benchmark/scripts/generate_demo_data.py
uv run bash benchmark/frameworks/trl/run.sh smoke

# Or activate
source .venv/bin/activate
python benchmark/tests/test_metrics_rewards.py
```

## Key files

| File | Role |
|---|---|
| `pyproject.toml` | Dependency declaration and extras |
| `uv.lock` | Locked resolution (commit this) |
| `.python-version` | Python version pin |
| `.venv/` | Local virtualenv (do not commit) |

## Adding dependencies

```bash
uv add <package>                  # main deps + update lock
uv add --optional trl <package>   # trl extra
uv lock                           # refresh lock only
uv sync --extra all               # sync into .venv
```

Legacy `requirements.txt` files only keep migration notes — do not use them as install entry points.
