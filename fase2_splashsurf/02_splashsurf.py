import argparse, subprocess, shutil, sys, os
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).parent))
from config import PLY_DIR, MESH_DIR, SURF_RADIUS, SURF_SMOOTH, SURF_CUBE_SIZE, SURF_THRESHOLD

try:
    from config import SURF_SMOOTH_ITERS as SMOOTH_ITERS
except ImportError:
    SMOOTH_ITERS = 25
try:
    from config import SURF_NORMAL_ITERS as NORMAL_ITERS
except ImportError:
    NORMAL_ITERS = 10


def check_splashsurf():
    if shutil.which('splashsurf') is None:
        print('[02] ERROR: splashsurf no esta en PATH.')
        print('     Rust: https://rustup.rs   ->   cargo install splashsurf')
        sys.exit(1)
    r = subprocess.run(['splashsurf', '--version'], capture_output=True, text=True)
    print(f'[02] {r.stdout.strip()}')


def reconstruct(job):
    src, dst, nthreads, radius, cube_size, smooth, threshold, smooth_iters, normal_iters = job
    cmd = [
        'splashsurf', 'reconstruct', str(src),
        f'--particle-radius={radius:.4f}',
        f'--smoothing-length={smooth:.4f}',   # multiplo del particle-radius (no metros)
        f'--cube-size={cube_size:.2f}',
        f'--surface-threshold={threshold:.2f}',
        '--mesh-smoothing-weights=on',
        f'--mesh-smoothing-iters={smooth_iters}',
        '--mesh-cleanup=on',
        '--normals=on',
        f'--normals-smoothing-iters={normal_iters}',
        f'--num-threads={nthreads}',
        f'--output-file={dst}',
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (src.name, r.returncode == 0, '' if r.returncode == 0 else r.stderr[-200:])


def main():
    ap = argparse.ArgumentParser(description='Reconstruccion paralela con splashsurf')
    ap.add_argument('--input',  default=str(PLY_DIR),  metavar='DIR')
    ap.add_argument('--output', default=str(MESH_DIR), metavar='DIR')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help='frames en paralelo (default: mitad de los cores)')
    ap.add_argument('--threads-per-job', type=int, default=2,
                    help='hilos de splashsurf por frame (default 2)')
    ap.add_argument('--radius', type=float, default=SURF_RADIUS,
                    help=f'radio de partícula para splashsurf; default {SURF_RADIUS:.4f}')
    ap.add_argument('--cube-size', type=float, default=SURF_CUBE_SIZE,
                    help='celda de marching cubes; 1.0 = fino (tomas de detalle), '
                         f'default {SURF_CUBE_SIZE} del config. OJO: memoria ~ (1.5/cube)^3')
    ap.add_argument('--smooth', type=float, default=SURF_SMOOTH,
                    help='smoothing-length (multiplo del radio); 2.3 = agua granular '
                         f'para camaras cercanas, default {SURF_SMOOTH}')
    ap.add_argument('--threshold', type=float, default=SURF_THRESHOLD,
                    help='iso-threshold de la superficie; 0.45 une agua continua, '
                         f'default {SURF_THRESHOLD}')
    ap.add_argument('--smooth-iters', type=int, default=SMOOTH_ITERS,
                    help=f'iteraciones de suavizado de malla (default {SMOOTH_ITERS})')
    ap.add_argument('--normal-iters', type=int, default=NORMAL_ITERS,
                    help=f'iteraciones de suavizado de normales (default {NORMAL_ITERS})')
    args = ap.parse_args()

    check_splashsurf()
    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(in_dir.glob('frame_*.ply'))
    if not frames:
        print(f'[02] No hay .ply en {in_dir}; ejecuta 07_extract_foam.py con --bulk-output')
        return

    jobs = [(src, out_dir / src.with_suffix('.obj').name, args.threads_per_job,
             args.radius, args.cube_size, args.smooth, args.threshold,
             args.smooth_iters, args.normal_iters)
            for src in frames
            if not (out_dir / src.with_suffix('.obj').name).exists()]

    print(f'[02] {len(frames)} frames, {len(jobs)} por reconstruir')
    print(f'     jobs={args.jobs}  x  {args.threads_per_job} hilos/frame  '
          f'(~{args.jobs * args.threads_per_job} cores)')
    print(f'     r={args.radius:.3f}  smooth={args.smooth}  cube={args.cube_size}  '
          f't={args.threshold}  iters={args.smooth_iters}/{args.normal_iters}')
    print(f'     {in_dir}  ->  {out_dir}')

    ok = fail = 0
    with Pool(args.jobs) as pool:
        for i, (name, success, err) in enumerate(pool.imap_unordered(reconstruct, jobs), 1):
            if success:
                ok += 1
            else:
                fail += 1
                print(f'     FAIL {name}: {err}')
            if i % 25 == 0 or i == len(jobs):
                print(f'     [{i}/{len(jobs)}]  ok={ok}  fail={fail}')

    print(f'[02] Done — {ok} reconstruidos, {fail} fallidos.')


if __name__ == '__main__':
    main()
