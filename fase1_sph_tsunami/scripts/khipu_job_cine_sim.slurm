#!/bin/bash
#SBATCH --job-name=sph_cine4m
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sph_cine_%j.out
#SBATCH --error=logs/sph_cine_%j.err


module purge
module load gnu12/12.4.0
module load cuda/12.8

WORK_DIR="$SLURM_SUBMIT_DIR"
BUILD_DIR="$WORK_DIR/build"
OUTPUT_DIR="$WORK_DIR/cache_cine"
SDF_PATH="$WORK_DIR/sdf/utec_sdf.bin"

N_FRAMES=1450
N_PARTICLES=4500000
SPACING_MULT=0.78
TARGET_FPS=125

mkdir -p "$OUTPUT_DIR" "$WORK_DIR/logs"

echo "=============================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Nodo:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "SDF:        $SDF_PATH"
echo "Frames:     $N_FRAMES @ target $TARGET_FPS fps"
echo "Particles:  cap=$N_PARTICLES spacing_mult=$SPACING_MULT"
echo "Output:     $OUTPUT_DIR"
echo "=============================="

[ -x "$BUILD_DIR/sph_tsunami" ] || { echo "ERROR: falta $BUILD_DIR/sph_tsunami"; exit 1; }
[ -f "$SDF_PATH" ] || { echo "ERROR: falta $SDF_PATH"; exit 1; }

"$BUILD_DIR/sph_tsunami" "$N_FRAMES" "$OUTPUT_DIR" \
    --sdf "$SDF_PATH" \
    --particles "$N_PARTICLES" \
    --spacing-mult "$SPACING_MULT" \
    --target-fps "$TARGET_FPS"

echo "Frames escritos: $(ls "$OUTPUT_DIR"/frame_*.bin 2>/dev/null | wc -l)"
du -sh "$OUTPUT_DIR"
echo "Fin: $(date)"
