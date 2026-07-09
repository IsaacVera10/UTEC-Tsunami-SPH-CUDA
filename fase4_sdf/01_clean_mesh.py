import argparse, sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
from config_v2 import GLB_INPUT, MESH_CLEAN_OUT, VOXEL_REMESH


def load_biggest_component(path: str) -> trimesh.Trimesh:
    print(f'[05] Cargando {path} ...')
    scene_or_mesh = trimesh.load(path, force='mesh', process=True)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.to_mesh()
    else:
        mesh = scene_or_mesh
    print(f'[05] Malla cruda: {len(mesh.vertices):,} verts, {len(mesh.faces):,} caras')

    comps = mesh.split(only_watertight=False)
    print(f'[05] Componentes conectados: {len(comps):,}')
    biggest = max(comps, key=lambda c: len(c.vertices))
    frac = len(biggest.vertices) / len(mesh.vertices) * 100
    print(f'[05] Componente dominante: {len(biggest.vertices):,} verts ({frac:.1f}%)')
    if frac < 90:
        print('[05] AVISO: el dominante tiene <90% de los vertices.')
        print('     Verifica que el input sea UTEC_RAW.glb y no el _texture.glb')
    return biggest


def voxel_remesh(mesh: trimesh.Trimesh, voxel: float) -> trimesh.Trimesh:
    import open3d as o3d
    print(f'[05] Voxel remesh @ {voxel} (unidades del GLB) ...')

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    vox = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_mesh, voxel_size=voxel)
    print(f'[05] Voxel grid: {len(vox.get_voxels()):,} voxels ocupados')

    from skimage import measure
    voxels = vox.get_voxels()
    idx = np.array([v.grid_index for v in voxels])
    dims = idx.max(axis=0) + 3                      # +2 de padding +1 inclusivo
    vol = np.zeros(dims, dtype=np.uint8)
    vol[idx[:, 0] + 1, idx[:, 1] + 1, idx[:, 2] + 1] = 1

    verts, faces, _, _ = measure.marching_cubes(vol.astype(np.float32), level=0.5)
    origin = np.asarray(vox.origin)
    verts_world = (verts - 1.0) * voxel + origin

    out = trimesh.Trimesh(vertices=verts_world, faces=faces, process=True)
    out.fix_normals()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  default=GLB_INPUT)
    ap.add_argument('--output', default=MESH_CLEAN_OUT)
    ap.add_argument('--voxel',  type=float, default=VOXEL_REMESH)
    ap.add_argument('--no-remesh', action='store_true',
                    help='solo componente dominante + fill_holes de trimesh '
                         '(mas fiel pero puede no cerrar del todo)')
    args = ap.parse_args()

    mesh = load_biggest_component(args.input)

    if args.no_remesh:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        clean = mesh
    else:
        clean = voxel_remesh(mesh, args.voxel)

    print(f'[05] Resultado: {len(clean.vertices):,} verts, '
          f'{len(clean.faces):,} caras, watertight={clean.is_watertight}')
    if not clean.is_watertight:
        print('[05] AVISO: aun no es watertight. Baja --voxel (mas fino) o')
        print('     sube (mas grueso, cierra mas agresivo) y reintenta.')

    clean.export(args.output)
    print(f'[05] Exportado -> {args.output}')
    bb = clean.bounds
    print(f'[05] Bbox (unidades GLB): min={bb[0].round(4)} max={bb[1].round(4)}')
    print('[05] Siguiente paso: python 06_generate_sdf.py')


if __name__ == '__main__':
    main()
