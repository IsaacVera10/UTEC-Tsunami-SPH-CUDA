import json, sys
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent))
from config_v2 import (UTEC_BBOX_MIN, UTEC_BBOX_MAX, SDF_MARGIN, SDF_VOXEL,
                       MESH_CLEAN_OUT, DOMAIN_MIN, DOMAIN_MAX)

OUT_DIR = Path(__file__).parent / 'sdf'


def place_in_sim_space(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Escala+traslada la malla (unidades GLB, Y-up) al bbox sim-space objetivo.

    El GLB es Y-up y el sim-space tambien usa Y=altura, asi que no hay
    rotacion: solo escala anisotropica al bbox objetivo y traslacion.
    (La rotacion X=90 de Blender es Blender Z-up vs glTF Y-up — no aplica aqui.)
    """
    src_min, src_max = mesh.bounds
    dst_min = np.array(UTEC_BBOX_MIN)
    dst_max = np.array(UTEC_BBOX_MAX)

    scale = (dst_max - dst_min) / (src_max - src_min)
    print(f'[06] Escala GLB->sim por eje: {scale.round(3)}  '
          f'(~{scale.mean():.1f} m/unidad promedio)')

    v = (mesh.vertices - src_min) * scale + dst_min
    placed = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)

    bb = placed.bounds
    print(f'[06] UTEC en sim-space: min={bb[0].round(2)} max={bb[1].round(2)}')
    dm_min, dm_max = np.array(DOMAIN_MIN), np.array(DOMAIN_MAX)
    if (bb[0] < dm_min - 1e-6).any() or (bb[1] > dm_max + 1e-6).any():
        print('[06] AVISO: el edificio sale del dominio de simulacion!')
    return placed


def bake_sdf(mesh: trimesh.Trimesh):
    grid_min = np.array(UTEC_BBOX_MIN) - SDF_MARGIN
    grid_max = np.array(UTEC_BBOX_MAX) + SDF_MARGIN
    dims = np.ceil((grid_max - grid_min) / SDF_VOXEL).astype(int) + 1
    print(f'[06] Grilla SDF: dims={dims} ({dims.prod():,} celdas) '
          f'voxel={SDF_VOXEL} m, origen={grid_min.round(2)}')

    xs = grid_min[0] + np.arange(dims[0]) * SDF_VOXEL
    ys = grid_min[1] + np.arange(dims[1]) * SDF_VOXEL
    zs = grid_min[2] + np.arange(dims[2]) * SDF_VOXEL
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)))))

    print('[06] Calculando distancias con signo (puede tardar ~1-2 min)...')
    d = scene.compute_signed_distance(
        o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32))
    sdf = d.numpy().reshape(dims)          # convencion o3d: negativo = dentro

    inside = (sdf < 0).sum()
    print(f'[06] Celdas dentro del edificio: {inside:,} '
          f'({inside / dims.prod() * 100:.1f}%)')
    return sdf, grid_min, dims


def export(sdf, grid_min, dims):
    OUT_DIR.mkdir(exist_ok=True)

    (OUT_DIR / 'utec_sdf.bin').write_bytes(sdf.astype(np.float32).tobytes())
    meta = {
        'origin':  [float(v) for v in grid_min],
        'dims':    [int(v) for v in dims],
        'voxel':   SDF_VOXEL,
        'order':   'x_slowest_z_fastest  (idx = (ix*dimY + iy)*dimZ + iz)',
        'sign':    'negative_inside',
    }
    (OUT_DIR / 'utec_sdf_meta.json').write_text(json.dumps(meta, indent=2))
    print(f'[06] SDF -> {OUT_DIR / "utec_sdf.bin"} '
          f'({sdf.nbytes / 1e6:.1f} MB) + utec_sdf_meta.json')

    from skimage import measure
    verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0)
    verts_world = verts * SDF_VOXEL + grid_min
    check = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    check.export(OUT_DIR / 'utec_sdf_check.obj')
    print(f'[06] Isosuperficie de control -> {OUT_DIR / "utec_sdf_check.obj"}')
    print('[06] VALIDACION: importa utec_sdf_check.obj en Blender junto al GLB')
    print('     original. Deben coincidir en forma (el check sera mas "gordo"')
    print('     por ~medio voxel, es normal). Si hay fugas/huecos -> volver al 05.')


def main():
    mesh_path = Path(__file__).parent / MESH_CLEAN_OUT
    if not mesh_path.exists():
        print(f'[06] No existe {mesh_path} — corre 05_clean_mesh.py primero')
        return
    mesh = trimesh.load(str(mesh_path), force='mesh')
    print(f'[06] Malla limpia: {len(mesh.vertices):,} verts, '
          f'watertight={mesh.is_watertight}')

    placed = place_in_sim_space(mesh)
    sdf, grid_min, dims = bake_sdf(placed)
    export(sdf, grid_min, dims)


if __name__ == '__main__':
    main()
