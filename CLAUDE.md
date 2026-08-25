# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FLAIR (`flair_t2i`): block-routed attribute control for MM-DiT text-to-image generation (Stable Diffusion 3.5-medium), targeting a CVPR 2027 submission. Solo-author project, Kaggle free-tier GPU only for actual generation — see `docs/superpowers/plans/2026-08-25-flair-master-roadmap.md` for the week-by-week plan and gates.

Architecture, call trees, and file-by-file walkthroughs already exist and should be read instead of re-derived:
- @docs/CODE_WALKTHROUGH.md — file map, what's built (Tasks 1-16b) vs. planned (17-22)
- @docs/EXECUTION_TREE.md — exact call trees and the core routing equation
- @docs/RUNBOOK.md — the 9-phase local→Kaggle→calibration workflow, troubleshooting table

## Testing

Tests run in Docker only — there is no local Python environment for this repo:
```
docker build -t flair-test .
./run-local.sh test        # docker run --rm flair-test python -m pytest -q
```
Expect `175 passed`. `./run-local.sh` also has `explain`, `summary`, and `shell` subcommands (see the script). Windows/Git-Bash rewrites `/app` to a Windows-style path unless `MSYS_NO_PATHCONV=1` is set — `run-local.sh` already sets this; don't strip it if editing that script.

No linter or formatter is configured (no ruff/black/eslint/etc.) — don't assume one when reviewing style.

## Git

- `docs/`, `notebooks/`, and `calibration_runs/` are listed in `.gitignore` but files under `docs/` and `notebooks/` are already tracked (force-added). New files added under these three paths will be silently ignored by plain `git add` — use `git add -f` for them. This matters for `calibration_runs/basm.npz` specifically: RUNBOOK Phase 9 instructs committing that one artifact despite the directory being gitignored.
- Commit directly to `main` — no branches/PRs in use.

## Workflow for new feature work

Tasks 1-16b were built via the superpowers workflow (write a plan with `writing-plans`, then implement with `test-driven-development`/`executing-plans`) — plans live in `docs/superpowers/plans/`, specs in `docs/superpowers/specs/`. Default to this same plan-then-TDD workflow for new feature work in this repo rather than editing ad hoc.
