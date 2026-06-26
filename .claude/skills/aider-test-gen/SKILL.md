---
name: aider-test-gen
description: Use when you need to generate large families of repetitive or parametrized pytest tests (tens of near-identical input/edge/parameter variants) for a target Python module, by driving aider with the repo's local Ollama model.
---

# aider-test-gen

Generate big, tedious-to-hand-write pytest suites — the kind with tens of
parametrized cases, many edge cases, and many parameter combinations — by
driving **aider** against a target source module. aider here runs on a **local
Ollama** backend (no cloud API key), configured via `OLLAMA_API_BASE` in the
repo's `.env`.

Use this when: a module needs broad test coverage and the tests are highly
repetitive (same call shape, many inputs). Do NOT use it for a couple of bespoke
tests — just write those by hand.

## Environment facts (this repo)

- aider version: **0.86.2** (already installed).
- Backend: **local Ollama**. `.env` contains `OLLAMA_API_BASE=http://127.0.0.1:11434`.
  There is no cloud API key.
- Default model: **`qwen2.5-coder:14b`** — a strong, reasonably fast local coder
  found via `ollama list`. For higher quality (slower) generation use
  **`qwen3-coder:30b`**. Other coders present: `qwen2.5-coder:7b`,
  `qwen2.5-coder:3b-base`. If `ollama list` shows none of these, run
  `ollama pull qwen2.5-coder:14b` first, or pass `--model <name>`.
- aider addresses Ollama models with the **`ollama_chat/`** provider prefix,
  e.g. `ollama_chat/qwen2.5-coder:14b`.
- Tests run with **`venv/bin/python -m pytest`**.

## Quick path: use the helper script

The script `gen_tests.sh` (next to this file) wraps everything: it loads `.env`,
defaults the model, marks the source read-only, marks the test file editable,
and runs aider non-interactively.

```bash
.claude/skills/aider-test-gen/gen_tests.sh \
  --source <path/to/source.py> \
  --out    <path/to/test_file.py> \
  --prompt "<what to test: cases, edge cases, parameter combos>" \
  [--model qwen3-coder:30b]   # optional, defaults to qwen2.5-coder:14b
```

Then verify (see "Verify" below). Re-run with a refined `--prompt` to iterate.

## How it works (the aider invocation)

The script runs this command. Use it directly if you need finer control:

```bash
aider \
  --model ollama_chat/qwen2.5-coder:14b \
  --read  <path/to/source.py> \      # read-only context: code under test
  --file  <path/to/test_file.py> \   # editable target: the test file
  --message "<instructions>" \       # one-shot, non-interactive, then exits
  --yes-always \                     # auto-confirm prompts
  --no-auto-commits \                # don't auto-commit
  --no-stream                        # cleaner scripted output
```

Flag notes (verified against aider 0.86.2):
- `--read FILE` — read-only context file. Pass the **source module** here so
  aider sees the code but never edits it.
- `--file FILE` — editable target. Pass the **test file** here.
- `--message`/`-m` — single instruction, runs once and exits (non-interactive).
- `--yes-always` — confirm everything (note: it's `--yes-always`, NOT `--yes`).
- `--no-auto-commits` — disable auto-commit (note the plural `commits`).
- `--no-stream` — optional, nicer for scripted/batch runs.
- `OLLAMA_API_BASE` must be in the environment; the script sources `.env` to set it.

## Step-by-step

1. **Pick the target module** and decide what to cover (input families,
   boundaries, empty/degenerate inputs, error cases, parameter grids).
2. **Choose the test file path**, e.g. `tests/test_<module>.py`.
3. **Run the helper** (or the raw aider command). In the `--prompt`, explicitly
   ask for `@pytest.mark.parametrize` with **tens of variants** — that is the
   whole point of this skill. Name the concrete edge cases you want.
4. **Verify** the tests run:
   ```bash
   venv/bin/python -m pytest <path/to/test_file.py> -q
   ```
5. **Iterate** if needed (see below).

## Writing a good prompt

Be specific and demand parametrization. Good prompt ingredients:
- "Use `@pytest.mark.parametrize` with at least 20-30 cases."
- Enumerate edge cases: empty input, single element, negative/zero, NaN/inf,
  very large values, wrong types.
- "Use `pytest.raises(...)` for invalid inputs."
- "Import the functions under test from `<source>`; file must run standalone."

## Iterating

If tests fail to collect, miss edge cases, or are thin, re-run aider on the SAME
test file (it's already populated, so aider refines it):

```bash
.claude/skills/aider-test-gen/gen_tests.sh \
  --source <source.py> --out <test_file.py> \
  --prompt "The current tests fail with: <paste the pytest error>. Fix the
            imports/assertions and ADD parametrized cases for NaN, empty input,
            and negative window sizes."
```

Repeat verify → refine until `pytest -q` is green and edge-case coverage is
adequate. If a model is too weak (bad imports, broken syntax), switch to a
stronger one: `--model qwen3-coder:30b`.

## Worked example

Generate a wide parametrized suite for a smoothing function:

```bash
.claude/skills/aider-test-gen/gen_tests.sh \
  --source webapp/src/detection/smoothing.py \
  --out    tests/test_smoothing_generated.py \
  --prompt "Generate a parametrized pytest suite for moving_average(). Use
            @pytest.mark.parametrize with 25+ cases: varied input arrays and
            window sizes, empty list, single element, window larger than input,
            window=1, NaN values, and negative window (expect ValueError via
            pytest.raises). Import from webapp/src/detection/smoothing.py."

venv/bin/python -m pytest tests/test_smoothing_generated.py -q
```

## Caveats

- Local model inference can be slow; the first call also loads the model into
  memory. Don't block indefinitely — let long runs finish in the background.
- Always **review** generated assertions: a local model can write tests that
  pass but assert the wrong behavior. Treat output as a draft to verify, not
  ground truth.
