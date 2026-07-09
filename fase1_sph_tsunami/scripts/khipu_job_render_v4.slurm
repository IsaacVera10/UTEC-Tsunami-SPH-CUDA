#!/bin/bash
#SBATCH --job-name=render_cine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/render_cine_%j.out
#SBATCH --error=logs/render_cine_%j.err


CAM="${1:?uso: sbatch khipu_job_render_v4.slurm <Camara> <mesh_dir> [start] [end] [stride] [foam_dir] [foam_radius]}"
MESH_SUBDIR="${2:?falta el dir de mallas}"
START="${3:-0}"
END="${4:-999999}"
STRIDE="${5:-1}"
FOAM_SUBDIR="${6:-}"    # dir con foam_XXXXXX.ply (del 07); vacio = sin espuma
FOAM_RADIUS="${7:-0.022}"   # radio de gota: 0.022 POV; 0.03 aerea sparse
ABSORPTION="${8:-}"     # override de Volume Absorption; aerea ~0.018 (0.045 = agua negra en tomas profundas)
OUT_SUFFIX="${9:-}"     # sufijo del dir de salida para variantes A/B/C sin colision de resume

module purge
module load gnu12/12.4.0

BLENDER="$HOME/blender-5.1.2-linux-x64/blender"
WORK_DIR="$SLURM_SUBMIT_DIR"
F2_DIR="$(dirname "$WORK_DIR")/fase2_splashsurf"
BLEND="$WORK_DIR/escena_utec_v4_cine.blend"
MESH_DIR="$WORK_DIR/$MESH_SUBDIR"
OUT_DIR="$WORK_DIR/render_v4_${CAM}${OUT_SUFFIX}"

mkdir -p "$WORK_DIR/logs" "$OUT_DIR"

echo "=============================="
echo "Job ID:  $SLURM_JOB_ID   Nodo: $SLURMD_NODENAME"
echo "Camara:  $CAM   mallas: $MESH_SUBDIR   rango: $START-$END stride=$STRIDE"
echo "Start:   $(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "=============================="

[ -x "$BLENDER" ] || { echo "ERROR: falta $BLENDER"; exit 1; }
[ -f "$BLEND" ]   || { echo "ERROR: falta $BLEND"; exit 1; }
[ -d "$MESH_DIR" ] || { echo "ERROR: falta $MESH_DIR"; exit 1; }

FOAM_ARGS=""
if [ -n "$FOAM_SUBDIR" ]; then
    case "$FOAM_SUBDIR" in
        /*) FOAM_DIR="$FOAM_SUBDIR" ;;              # ruta absoluta tal cual
        *)  FOAM_DIR="$WORK_DIR/$FOAM_SUBDIR" ;;    # relativa al submit dir
    esac
    [ -d "$FOAM_DIR" ] || { echo "ERROR: falta $FOAM_DIR"; exit 1; }
    FOAM_ARGS="--foam-dir $FOAM_DIR --foam-radius $FOAM_RADIUS"
    echo "Espuma:  $FOAM_DIR (radio $FOAM_RADIUS)"
fi

ABS_ARGS=""
[ -n "$ABSORPTION" ] && { ABS_ARGS="--absorption $ABSORPTION"; echo "Absorcion: $ABSORPTION"; }

"$BLENDER" -b "$BLEND" -P "$F2_DIR/04_render_sequence.py" -- \
    --meshes "$MESH_DIR" --out "$OUT_DIR" \
    --camera "$CAM" --start "$START" --end "$END" --stride "$STRIDE" \
    --samples 128 --bounces 8 $FOAM_ARGS $ABS_ARGS

N_PNG=$(ls "$OUT_DIR"/*.png 2>/dev/null | wc -l)
echo "PNGs en $OUT_DIR: $N_PNG"
echo "Fin: $(date)"
