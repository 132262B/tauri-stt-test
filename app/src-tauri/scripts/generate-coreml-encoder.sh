#!/usr/bin/env bash
set -euo pipefail

# Build-time/dev-only helper. Do not run this from the app, and do not ship
# Python with iOS. The iOS runtime should only bundle the generated .mlmodelc.

MODEL="${1:-large-v3-turbo}"
COREML_QUANTIZE="${COREML_QUANTIZE:-True}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_TAURI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SRC_TAURI_DIR}/../.." && pwd)"
WORK_DIR="${REPO_ROOT}/.cache/whisper.cpp"
VENV_DIR="${REPO_ROOT}/.cache/coreml-venv"
OUT_DIR="${SRC_TAURI_DIR}/models/ggml"
EXPECTED="${OUT_DIR}/ggml-${MODEL}-encoder.mlmodelc"

if [[ "${MODEL}" == *q5* ]]; then
  echo "Pass the unquantized model id, e.g. large-v3-turbo, not ${MODEL}" >&2
  exit 2
fi

for cmd in git xcrun; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 127
  fi
done

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "${candidate}")"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Missing required command: python3.12/python3.11/python3" >&2
  exit 127
fi

if ! xcrun --find coremlc >/dev/null 2>&1; then
  echo "Xcode coremlcompiler is required. Install/select full Xcode, then run xcode-select." >&2
  exit 127
fi

mkdir -p "$(dirname "${WORK_DIR}")" "${OUT_DIR}"

if [[ ! -d "${WORK_DIR}/.git" ]]; then
  git clone https://github.com/ggml-org/whisper.cpp.git "${WORK_DIR}"
else
  git -C "${WORK_DIR}" fetch --depth 1 origin
  git -C "${WORK_DIR}" checkout origin/master
fi

if [[ ! -x "${VENV_DIR}/bin/python3" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python3" -m pip install --upgrade pip
"${VENV_DIR}/bin/python3" -m pip install -r "${WORK_DIR}/models/requirements-coreml.txt"
CERT_FILE="$("${VENV_DIR}/bin/python3" -c 'import certifi; print(certifi.where())')"

(
  cd "${WORK_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  export SSL_CERT_FILE="${CERT_FILE}"
  export REQUESTS_CA_BUNDLE="${CERT_FILE}"
  python3 models/convert-whisper-to-coreml.py \
    --model "${MODEL}" \
    --encoder-only True \
    --optimize-ane True \
    --quantize "${COREML_QUANTIZE}"
  xcrun coremlc compile "models/coreml-encoder-${MODEL}.mlpackage" models/
  rm -rf "models/ggml-${MODEL}-encoder.mlmodelc"
  mv -v "models/coreml-encoder-${MODEL}.mlmodelc" "models/ggml-${MODEL}-encoder.mlmodelc"
)

candidate="$(find "${WORK_DIR}" -type d -name "ggml-${MODEL}-encoder.mlmodelc" -print -quit)"

if [[ -z "${candidate}" ]]; then
  echo "CoreML encoder was not generated for ${MODEL}" >&2
  exit 1
fi

rm -rf "${EXPECTED}"
cp -R "${candidate}" "${EXPECTED}"
echo "Wrote ${EXPECTED}"
