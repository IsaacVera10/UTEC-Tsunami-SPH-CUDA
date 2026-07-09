import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def parse_args():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:] if '--' in argv else argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True, help='dir con frame_XXXXXX.bin')
    ap.add_argument('--output', required=True, help='dir de salida foam_XXXXXX.ply')
    ap.add_argument('--start',  type=int, default=0)
    ap.add_argument('--end',    type=int, default=10**9)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--spacing', type=float, default=0.135,
                    help='spacing de la sim (v2: 0.135, v4: 0.105) — define la celda')
    ap.add_argument('--cell-mult', type=float, default=2.2,
                    help='celda = spacing * cell_mult')
    ap.add_argument('--max-box-neighbors', type=int, default=75,
                    help='umbral de spray en conteo-caja 3x3x3 (default 75 ~ '
                         '<=12 vecinos exactos en esfera; independiente de la '
                         'resolucion porque el conteo escala con celda^3*densidad). '
                         'Pasar 0 para usar --spray-pct')
    ap.add_argument('--spray-pct', type=float, default=10.0,
                    help='solo si --max-box-neighbors 0: %% mas aislado = spray')
    ap.add_argument('--max-points', type=int, default=400000,
                    help='tope de puntos de foam por frame. Los puntos son point '
                         'cloud nativo de Cycles (baratos), no esferas instanciadas')
    ap.add_argument('--min-y', type=float, default=0.35,
                    help='spray por debajo de esta altura sim va al bulk (piso)')
    ap.add_argument('--wall-margin', type=float, default=1.0,
                    help='spray a menos de esto de las paredes del dominio va al '
                         'bulk (evita la corona blanca en los bordes)')
    ap.add_argument('--domain', type=float, nargs=6,
                    default=[0.0, 0.0, 0.0, 60.0, 30.0, 30.0],
                    help='xmin ymin zmin xmax ymax zmax del dominio sim')
    ap.add_argument('--bulk-output', default=None,
                    help='dir para frame_XXXXXX.ply del BULK (agua sin spray). '
                         'CLAVE anti "ojo de pez": splashsurf debe mallar este PLY '
                         'en vez del cache crudo, asi el spray solo existe como foam')
    ap.add_argument('--jobs', type=int, default=1, help='frames en paralelo')
    return ap.parse_args(argv)


def read_bin(path: Path):
    raw = path.read_bytes()
    n = struct.unpack('<I', raw[:4])[0]
    pos = np.frombuffer(raw, dtype=np.float32, count=n * 3, offset=4).reshape(n, 3)
    return pos


def box_neighbor_counts(pos: np.ndarray, cell: float) -> np.ndarray:
    """Nº de partículas en la caja 3x3x3 de celdas alrededor de cada partícula."""
    ijk = np.floor(pos / cell).astype(np.int64)
    ijk -= ijk.min(axis=0)  # no-negativos para empaquetar
    dims = ijk.max(axis=0).astype(np.int64) + 3  # +margen para offsets ±1
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = np.argsort(key, kind='stable')
    skey = key[order]
    ukey, counts = np.unique(skey, return_counts=True)
    cum = np.zeros(len(ukey) + 1, dtype=np.int64)
    np.cumsum(counts, out=cum[1:])
    total = np.zeros(len(pos), dtype=np.int32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                off = (dx * dims[1] + dy) * dims[2] + dz
                idx = np.searchsorted(ukey, key + off)
                idx_clip = np.minimum(idx, len(ukey) - 1)
                hit = ukey[idx_clip] == key + off
                total += np.where(hit, counts[idx_clip], 0).astype(np.int32)
    return total


def write_ply(path: Path, pts: np.ndarray):
    hdr = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n'
           'property float x\nproperty float y\nproperty float z\nend_header\n')
    with open(path, 'wb') as f:
        f.write(hdr.encode('ascii'))
        f.write(np.ascontiguousarray(pts, dtype=np.float32).tobytes())


def process_frame(job):
    src, dst, bulk_dst, args = job
    pos = read_bin(src)
    cell = args.spacing * args.cell_mult
    cnt = box_neighbor_counts(pos, cell)

    if args.max_box_neighbors > 0:
        thr = args.max_box_neighbors
    else:
        thr = int(np.percentile(cnt, args.spray_pct))

    d = args.domain
    m = args.wall_margin
    inside = ((pos[:, 0] > d[0] + m) & (pos[:, 0] < d[3] - m) &
              (pos[:, 2] > d[2] + m) & (pos[:, 2] < d[5] - m))
    foam_mask = (cnt <= thr) & (pos[:, 1] > args.min_y) & inside
    idx = np.flatnonzero(foam_mask)

    if len(idx) > args.max_points:
        sc = cnt[idx].astype(np.float64) - 0.001 * pos[idx, 1]
        keep = np.argsort(sc, kind='stable')[:args.max_points]
        idx = idx[keep]

    bulk_mask = ~foam_mask
    write_ply(dst, pos[idx])
    if bulk_dst is not None:
        write_ply(bulk_dst, pos[bulk_mask])
    return (f'{src.name}: N={len(pos):,} thr={thr} '
            f'foam={len(idx):,} bulk={int(bulk_mask.sum()):,}')


def main():
    args = parse_args()
    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        print(f'[07] ERROR: no existe {in_dir.resolve()} (usa ruta ABSOLUTA: '
              'Blender/SLURM no siempre heredan el cwd esperado)')
        return

    bulk_dir = None
    if args.bulk_output:
        bulk_dir = Path(args.bulk_output)
        bulk_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for f in sorted(in_dir.glob('frame_*.bin')):
        i = int(f.stem.split('_')[1])
        if not (args.start <= i <= args.end) or (i - args.start) % args.stride:
            continue
        dst = out_dir / f'foam_{i:06d}.ply'
        bulk_dst = (bulk_dir / f'frame_{i:06d}.ply') if bulk_dir else None
        done_foam = dst.exists()
        done_bulk = bulk_dst is None or bulk_dst.exists()
        if done_foam and done_bulk:
            continue
        jobs.append((f, dst, bulk_dst, args))

    print(f'[07] {len(jobs)} frames por extraer -> {out_dir}  '
          f'(spacing={args.spacing}, celda={args.spacing*args.cell_mult:.3f}, '
          f'max_points={args.max_points})')

    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            for k, msg in enumerate(pool.imap_unordered(process_frame, jobs)):
                if k % 25 == 0:
                    print(f'  [{k+1}/{len(jobs)}] {msg}', flush=True)
    else:
        for k, job in enumerate(jobs):
            msg = process_frame(job)
            if k % 25 == 0:
                print(f'  [{k+1}/{len(jobs)}] {msg}', flush=True)
    print('[07] Done.')


if __name__ == '__main__':
    main()
