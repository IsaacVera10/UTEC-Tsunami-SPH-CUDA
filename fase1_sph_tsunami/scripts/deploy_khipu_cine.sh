#!/bin/bash
set -euo pipefail

REMOTE="${1:?Uso: bash deploy_khipu_cine.sh usuario@khipu.utec.edu.pe}"
REMOTE_ROOT="${2:-computer_graphics}"
REMOTE_FASE1="$REMOTE_ROOT/fase1_sph_tsunami"
REMOTE_FASE2="$REMOTE_ROOT/fase2_splashsurf"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FASE1="$(dirname "$SCRIPT_DIR")"
ROOT="$(dirname "$FASE1")"
FASE2="$ROOT/fase2_splashsurf"
FASE3="$ROOT/fase3"
SDF="$ROOT/fase4_sdf/sdf"

FILES=(
    "$FASE1/CMakeLists.txt"
    "$FASE1/src/main.cu"
    "$FASE1/src/kernels.cu"
    "$FASE1/include/sph_types.h"
    "$FASE1/include/spatial_hash.cuh"
    "$FASE1/scripts/compile_login.sh"
    "$FASE1/scripts/khipu_job_cine_sim.slurm"
    "$FASE1/scripts/khipu_job_cine_splash_range.slurm"
    "$FASE1/scripts/khipu_job_render_v4.slurm"
    "$FASE2/config.py"
    "$FASE2/02_splashsurf.py"
    "$FASE2/04_render_sequence.py"
    "$FASE2/06_setup_render_scene_v4_cine.py"
    "$FASE2/07_extract_foam.py"
    "$SDF/utec_sdf.bin"
    "$SDF/utec_sdf_meta.json"
)

for file in "${FILES[@]}"; do
    [ -f "$file" ] || { echo "FALTA: $file"; exit 1; }
done

ssh "$REMOTE" "mkdir -p \
    $REMOTE_FASE1/src \
    $REMOTE_FASE1/include \
    $REMOTE_FASE1/scripts \
    $REMOTE_FASE1/sdf \
    $REMOTE_FASE1/logs \
    $REMOTE_FASE2"

scp "$FASE1/CMakeLists.txt" "$REMOTE:$REMOTE_FASE1/"
scp "$FASE1/src/main.cu" "$FASE1/src/kernels.cu" "$REMOTE:$REMOTE_FASE1/src/"
scp "$FASE1/include/sph_types.h" "$FASE1/include/spatial_hash.cuh" "$REMOTE:$REMOTE_FASE1/include/"
scp "$FASE1/scripts/"*.sh "$FASE1/scripts/"*.slurm "$REMOTE:$REMOTE_FASE1/scripts/"
scp "$FASE2/"*.py "$REMOTE:$REMOTE_FASE2/"
scp "$SDF/utec_sdf.bin" "$SDF/utec_sdf_meta.json" "$REMOTE:$REMOTE_FASE1/sdf/"

if [ -f "$FASE3/escena_utec_v4_cine.blend" ]; then
    scp "$FASE3/escena_utec_v4_cine.blend" "$REMOTE:$REMOTE_FASE1/"
fi

ssh "$REMOTE" "sed -i 's/\r\$//' \
    $REMOTE_FASE1/scripts/*.sh \
    $REMOTE_FASE1/scripts/*.slurm \
    $REMOTE_FASE2/*.py"

echo "Deploy completo: $REMOTE:$REMOTE_ROOT"
