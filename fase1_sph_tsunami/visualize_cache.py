import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct, glob, sys

def read_frame(path):
    with open(path, 'rb') as f:
        n = struct.unpack('I', f.read(4))[0]
        data = np.frombuffer(f.read(n * 12), dtype=np.float32).reshape(n, 3)
    return data

import sys, os

if len(sys.argv) > 1:
    FRAME_DIR = sys.argv[1]
elif os.path.isdir('./output'):
    FRAME_DIR = './output'
else:
    FRAME_DIR = './build/test_output'

frames = sorted(glob.glob(f'{FRAME_DIR}/frame_*.bin'))
print(f"Frames encontrados: {len(frames)}  [{FRAME_DIR}]")

XMAX, YMAX, ZMAX = 60.0, 15.0, 30.0
DT = 0.001
STEPS_PER_FRAME = 17

n = len(frames)
checkpoints = sorted(set([0, n//4, n//2, max(0, n-1)]))
checkpoints = [i for i in checkpoints if i < n]

if frames:
    _mx = np.zeros(3)
    for _i in checkpoints:
        _mx = np.maximum(_mx, read_frame(frames[_i]).max(axis=0))
    XMAX = float(np.ceil(_mx[0] / 10) * 10)
    YMAX = float(np.ceil(_mx[1] / 5)  * 5)
    ZMAX = float(np.ceil(_mx[2] / 5)  * 5)

Z_MID   = ZMAX / 2
Z_SLICE = max(1.0, ZMAX * 0.07)

for idx in checkpoints:
    if idx >= len(frames): continue
    pts = read_frame(frames[idx])

    mask = np.abs(pts[:,2] - Z_MID) < Z_SLICE / 2
    sl   = pts[mask]

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    if len(sl) > 0:
        ax1.scatter(sl[:,0], sl[:,1], s=8, c=sl[:,1], cmap='Blues',
                    vmin=0, vmax=YMAX, alpha=0.8)
    ax1.set_xlim(0, XMAX); ax1.set_ylim(0, YMAX)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y altura (m)')
    ax1.set_title(f'Sección z=[{Z_MID-Z_SLICE/2:.1f},{Z_MID+Z_SLICE/2:.1f}]m  ({len(sl)} part.)')
    ax1.axvline(XMAX/3, color='red', ls='--', lw=0.8, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(pts[:,0], pts[:,2], pts[:,1], s=1, c=pts[:,1], cmap='Blues',
                vmin=0, vmax=YMAX)
    ax2.set_xlim(0, XMAX); ax2.set_ylim(0, ZMAX); ax2.set_zlim(0, YMAX)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)'); ax2.set_zlabel('Y (m)')
    ax2.set_title('Vista 3D')

    sim_t = idx * STEPS_PER_FRAME * DT
    fig.suptitle(f'Frame {idx} (t={sim_t:.3f}s) — {len(pts):,} partículas', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'frame_{idx:04d}.png', dpi=100)
    plt.close()
    print(f'Guardado frame_{idx:04d}.png  x=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]')
