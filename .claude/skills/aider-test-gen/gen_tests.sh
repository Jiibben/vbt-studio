#!/usr/bin/env bash
#
# gen_tests.sh — drive aider (with a local Ollama model) to generate large
# families of repetitive / parametrized pytest tests for a target source module.
#
# Usage:
#   gen_tests.sh --source <src.py> --out <test_file.py> --prompt "<what to test>" [--model <ollama_model>]
#   gen_tests.sh --help
#
# Notes:
#   * The source file is passed to aider as READ-ONLY context (--read) so aider
#     can see the code under test but will not modify it.
#   * The output test file is the single EDITABLE target (--file).
#   * Auto-commits are disabled so nothing is committed automatically.
#   * .env is loaded so OLLAMA_API_BASE is available to aider/litellm.
#
set -euo pipefail

# --- defaults ---------------------------------------------------------------
# Default model discovered via `ollama list` on this machine. qwen2.5-coder:14b
# is a strong, reasonably fast local coding model. For higher-quality (slower)
# generation, pass --model ollama_chat/qwen3-coder:30b.
DEFAULT_MODEL="qwen2.5-coder:14b"

SOURCE=""
OUT=""
PROMPT=""
MODEL=""

usage() {
  cat <<'EOF'
gen_tests.sh — generate parametrized pytest tests via aider + local Ollama

Required:
  --source <path>   Source module under test (passed read-only to aider)
  --out <path>      Test file to create/edit (the editable target)
  --prompt "<text>" Description of what to test (cases, edge cases, params)

Optional:
  --model <name>    Ollama model name WITHOUT the ollama_chat/ prefix
                    (default: qwen2.5-coder:14b)
  --help            Show this help and exit

Example:
  ./gen_tests.sh \
    --source webapp/src/detection/smoothing.py \
    --out tests/test_smoothing.py \
    --prompt "Generate a parametrized pytest suite covering many input arrays,
              window sizes, empty/single-element inputs, NaN handling and
              boundary conditions. Prefer @pytest.mark.parametrize with tens of
              cases."
EOF
}

# --- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="${2:-}"; shift 2 ;;
    --out)    OUT="${2:-}";    shift 2 ;;
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --model)  MODEL="${2:-}";  shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# --- validate ---------------------------------------------------------------
if [[ -z "$SOURCE" || -z "$OUT" || -z "$PROMPT" ]]; then
  echo "ERROR: --source, --out and --prompt are all required." >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$SOURCE" ]]; then
  echo "ERROR: source file not found: $SOURCE" >&2
  exit 2
fi

# --- locate repo root & load .env ------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  # Export every assignment in .env (e.g. OLLAMA_API_BASE) into this shell.
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
# Fallback if .env didn't set it.
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://127.0.0.1:11434}"

# --- pick model -------------------------------------------------------------
MODEL="${MODEL:-$DEFAULT_MODEL}"
# aider/litellm address Ollama models with the ollama_chat/ provider prefix.
AIDER_MODEL="ollama_chat/${MODEL}"

# Ensure the output file exists so aider treats it as an editable target.
mkdir -p "$(dirname "$OUT")"
[[ -f "$OUT" ]] || : > "$OUT"

echo "Source (read-only): $SOURCE"
echo "Target (editable):  $OUT"
echo "Model:              $AIDER_MODEL"
echo "OLLAMA_API_BASE:    $OLLAMA_API_BASE"
echo

# --- compose the instruction sent to aider ----------------------------------
# We steer aider toward many parametrized variants and a runnable pytest file.
FULL_MSG="You are writing pytest tests in ${OUT} for the code in ${SOURCE} \
(provided read-only). ${PROMPT}

Requirements:
- Use pytest. Import the module/functions under test from ${SOURCE}.
- Generate MANY cases. Strongly prefer @pytest.mark.parametrize with tens of
  near-identical variants rather than copy-pasted test functions.
- Cover normal inputs, boundary values, empty/degenerate inputs, and error
  cases (use pytest.raises where appropriate).
- The file must be self-contained and runnable with:
  venv/bin/python -m pytest ${OUT} -q
- Do not modify ${SOURCE}; only edit ${OUT}."

# --- run aider --------------------------------------------------------------
# Flags verified against aider 0.86.2:
#   --model         main chat model
#   --read FILE     read-only context file (source under test)
#   --file FILE     editable target (the test file)
#   --message/-m    one-shot non-interactive instruction, then exit
#   --yes-always    auto-confirm all prompts (non-interactive)
#   --no-auto-commits  do not auto-commit changes
#   --no-stream     simpler/cleaner output for scripted runs
exec aider \
  --model "$AIDER_MODEL" \
  --read "$SOURCE" \
  --file "$OUT" \
  --message "$FULL_MSG" \
  --yes-always \
  --no-auto-commits \
  --no-stream
