#!/bin/sh
set -eu
BASE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TARGET=${1:?Usage: sh reproduce.sh /absolute/new/run}
PYTHON_BIN=${PYTHON_BIN:-/home/fangxy/.conda/envs/nas/bin/python}
if [ -e "$TARGET" ]; then
    echo "Target must not already exist; preserve earlier evidence." >&2
    exit 2
fi
mkdir -p "$TARGET"
cp -R "$BASE/code" "$TARGET/code"
for name in stage1_protocol.json stage1_protocol.sha256 selected_data.json selection_manifest.json data_access_audit.json; do
    cp "$BASE/$name" "$TARGET/$name"
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PYTHON_BIN" "$TARGET/code/run_stage1.py" synthetic --root "$TARGET"
"$PYTHON_BIN" "$TARGET/code/run_stage1.py" real --root "$TARGET"
"$PYTHON_BIN" "$TARGET/code/verify_artifacts.py" --root "$TARGET"
