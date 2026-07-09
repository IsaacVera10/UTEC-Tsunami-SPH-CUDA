#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$WORK_DIR/build"

echo "=== Compilando sph_tsunami en nodo login ==="
echo "Fuentes: $WORK_DIR"
echo "Build:   $BUILD_DIR"

module purge
module load gnu12/12.4.0
module load cuda/12.8
module load cmake/3.24.2

export CPATH="/usr/include:${CPATH}"
export C_INCLUDE_PATH="/usr/include:${C_INCLUDE_PATH}"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$WORK_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80"

cmake --build . --target sph_tsunami -j "$(nproc)"

echo ""
echo "=== Compilación exitosa: $BUILD_DIR/sph_tsunami ==="
echo "Ahora puedes enviar el job:"
echo "  cd $WORK_DIR && sbatch scripts/khipu_job.slurm"
