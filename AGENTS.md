## Development Workflow

- Always use `uv run` instead of raw `python` when invoking project tooling.
- Before committing, run:
  - `make format`
  - `make type`
  - `make test` when the change is broad or user-facing
- Prefer targeted tests during iteration:
  - `uv run pytest tests/<file>.py`
  - `uv run pytest tests/`
- For lint/format/type checks:
  - `uv run ruff format`
  - `uv run ruff check --fix`
  - `uv run ty check`
  - `uv run pyright`

## Project Map

- `pyproject.toml`: source of truth for console scripts, dependency groups, and `uv` behavior
- `src/mjlab/`: primary application code
- `src/mjlab/scripts/`: user-facing CLI entry points for training, play, export, motion processing, and debugging
- `src/mjlab/tasks/`: task registration and task-specific environment / RL configs
- `tests/`: regression and integration coverage; add focused tests for new task variants and CLI wrappers
- `docs/`: user and developer docs
- `data/`: local data assets used by docs, examples, and motion workflows
- `artifacts/`, `logs/`, `wandb/`: runtime outputs; do not commit generated contents

## Local Runtime Content

- Keep local tool state and generated dependency checkouts out of version control:
  - `.codex/`
  - `.omx/`
  - `.external/`
  - temporary stray files like `=0.6.1`
- Treat these as machine-local working directories, not source files.

## Commit Guidance

- Use concise commits with a clear “why” in the title.
- Preserve the baseline task behavior when adding specialized variants.
- For motion / tracking work, prefer adding new task IDs over mutating existing defaults.
- Do not commit code that fails `ruff` or relevant `pytest` coverage.
