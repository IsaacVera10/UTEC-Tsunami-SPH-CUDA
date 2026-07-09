#!/bin/bash
#SBATCH --job-name=splash_cine
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH --output=logs/splash_cine_%j.out
#SBATCH --error=logs/splash_cine_%j.err


MODE="${1:-continuous}"
START="${2:-0}"
END="${3:-900}"

module purge
module load gnu12/12.4.0
module load python3/3.11.11
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"

WORK_DIR="$SLURM_SUBMIT_DIR"
F2_DIR="$(dirname "$WORK_DIR")/fase2_splashsurf"
CACHE_DIR="$WORK_DIR/cache_cine"

case "$MODE" in
    continuous)
        RADIUS=0.115
        SMOOTH=3.2
        CUBE=2.5
        THRESHOLD=0.50
        SMOOTH_ITERS=30
        NORMAL_ITERS=15
        JOBS=1
        THREADS=32
        ;;
    granular)
        RADIUS=0.105
        SMOOTH=2.6
        CUBE=1.0
        THRESHOLD=0.55
        SMOOTH_ITERS=15
        NORMAL_ITERS=8
        JOBS=2
        THREADS=16
        ;;
    *)
        echo "ERROR: MODE debe ser continuous o granular (recibido: $MODE)"
        exit 1
        ;;
esac

TAG="${MODE}_${START}_${END}"
BIN_LINKS="$WORK_DIR/binlinks_cine_$TAG"
PLY_DIR="$WORK_DIR/ply_cine_$TAG"
MESH_DIR="$WORK_DIR/mesh_cine_$TAG"
RECIPE="--radius $RADIUS --cube-size $CUBE --smooth $SMOOTH --threshold $THRESHOLD --smooth-iters $SMOOTH_ITERS --normal-iters $NORMAL_ITERS"

mkdir -p "$WORK_DIR/logs"
rm -rf "$BIN_LINKS"
mkdir -p "$BIN_LINKS"

echo "=============================="
echo "Job ID:  $SLURM_JOB_ID   Nodo: $SLURMD_NODENAME"
echo "Modo:    $MODE   rango: $START-$END"
echo "Receta:  $RECIPE"
echo "Start:   $(date)"
echo "=============================="

[ -d "$CACHE_DIR" ] || { echo "ERROR: falta $CACHE_DIR"; exit 1; }
[ -f "$F2_DIR/02_splashsurf.py" ] || { echo "ERROR: falta $F2_DIR/02_splashsurf.py"; exit 1; }

echo "--- Linkeando .bin del rango ---"
for ((i=START; i<=END; i++)); do
    f=$(printf "frame_%06d.bin" "$i")
    [ -f "$CACHE_DIR/$f" ] || { echo "ERROR: falta $CACHE_DIR/$f"; exit 1; }
    ln -s "$CACHE_DIR/$f" "$BIN_LINKS/$f"
done
echo "Links: $(ls "$BIN_LINKS"/frame_*.bin | wc -l)"

echo "--- Paso 07: bin -> bulk ply + foam ---"
python3 "$F2_DIR/07_extract_foam.py" --input "$BIN_LINKS" \
    --bulk-output "$PLY_DIR" --output "$WORK_DIR/foam_cine" \
    --spacing 0.105 --max-box-neighbors 40 --min-y 1.6 --jobs 8

echo "--- Paso 02: splashsurf $MODE ($JOBS jobs x $THREADS hilos) ---"
python3 "$F2_DIR/02_splashsurf.py" --input "$PLY_DIR" --output "$MESH_DIR" \
    $RECIPE --jobs "$JOBS" --threads-per-job "$THREADS"

echo "--- Paso 02b: barrido secuencial de fallidos ---"
python3 "$F2_DIR/02_splashsurf.py" --input "$PLY_DIR" --output "$MESH_DIR" \
    $RECIPE --jobs 1 --threads-per-job 32

N_BIN=$(ls "$BIN_LINKS"/frame_*.bin | wc -l)
N_OBJ=$(ls "$MESH_DIR"/*.obj 2>/dev/null | wc -l)
echo "OBJs cine generados: $N_OBJ / $N_BIN"
du -sh "$MESH_DIR"

rm -rf "$PLY_DIR" "$BIN_LINKS"
echo "Fin: $(date)"
