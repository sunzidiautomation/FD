---
name: flair-test
description: Run the FLAIR test suite via Docker (the only supported way to run tests locally) and interpret the result. Use when the user asks to run tests, verify changes, or check that the suite passes for this repo.
---

Run the project's Docker-based test loop (see `docs/RUNBOOK.md` Phase 0):

```
docker build -t flair-test .
./run-local.sh test
```

`./run-local.sh test` runs `docker run --rm flair-test python -m pytest -q` inside the image.

Expected output: `175 passed`.

If it fails, check for these known causes before treating it as a real regression:
- **Fewer than 175 tests collected / import errors**: the Docker image may be stale — rebuild with `docker build -t flair-test .` (no cache flag needed unless dependencies changed; use `--no-cache` if a `requirements.txt` change isn't picking up).
- **Path errors mentioning something other than `/app/...`**: `MSYS_NO_PATHCONV` wasn't set — this should not happen since `run-local.sh` sets it internally, but if invoking `docker run` directly on Windows/Git-Bash, prefix with `MSYS_NO_PATHCONV=1`.
- **Real failures**: report them normally — do not silently rerun or suppress.

Do not attempt to run `pytest` directly on the host — there is no local Python environment set up for this repo; Docker is the only supported path.
