# 使用 uv 管理 Python 开发环境

本仓库统一用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本、虚拟环境和依赖锁定。

## 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

## 一键同步

```bash
# 推荐：安装全部 extras + dev 工具
bash scripts/setup_uv.sh all

# 或按框架
bash scripts/setup_uv.sh trl
bash scripts/setup_uv.sh openrlhf
bash scripts/setup_uv.sh verl
```

等价命令：

```bash
uv python install   # 读取 .python-version（默认 3.12）
uv sync             # 基础依赖
uv sync --extra trl
uv sync --extra openrlhf
uv sync --extra all --group dev
```

## 日常使用

```bash
# 不激活 venv，直接跑
uv run python benchmark/scripts/generate_demo_data.py
uv run bash benchmark/frameworks/trl/run.sh smoke

# 或激活后使用
source .venv/bin/activate
python benchmark/tests/test_metrics_rewards.py
```

## 关键文件

| 文件 | 作用 |
|---|---|
| `pyproject.toml` | 依赖声明与 extras |
| `uv.lock` | 锁定解析结果（需提交） |
| `.python-version` | Python 版本钉扎 |
| `.venv/` | 本地虚拟环境（不提交） |

## 添加依赖

```bash
uv add <package>                  # 写入主依赖并更新 lock
uv add --optional trl <package>   # 写入 trl extra
uv lock                           # 仅刷新 lock
uv sync --extra all               # 同步到 .venv
```

旧的 `requirements.txt` 仅保留迁移提示，请勿再作为安装入口。
