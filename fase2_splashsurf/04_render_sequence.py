import bpy, sys, argparse
from pathlib import Path

try:
    import numpy as np          # el Blender oficial lo trae; guard por si acaso
except ImportError:
    np = None


def parse_args():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:] if '--' in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--meshes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=10**9)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--samples', type=int, default=None)
    ap.add_argument('--absorption', type=float, default=None,
                    help='override densidad de Volume Absorption del material Water')
    ap.add_argument('--bounces', type=int, default=None)
    ap.add_argument('--camera', default=None,
                    help='nombre de la camara a usar (Cam_Aerea / Cam_Azotea / Cam_Detalle); '
                         'default: la activa del .blend')
    ap.add_argument('--foam-dir', default=None,
                    help='dir con foam_XXXXXX.ply (del 07). Opcional: sin flag, '
                         'render identico al de siempre')
    ap.add_argument('--foam-radius', type=float, default=0.022,
                    help='radio de cada punto de espuma en unidades sim (OJO: la '
                         'escala del objeto agua ~2.5x lo amplifica en mundo; '
                         '0.022 sim ~ 11 cm de gota en pantalla)')
    return ap.parse_args(argv)


def read_foam_ply(path):
    raw = Path(path).read_bytes()
    end = raw.find(b'end_header\n') + len(b'end_header\n')
    n = 0
    for line in raw[:end].decode('ascii', 'replace').splitlines():
        if line.startswith('element vertex'):
            n = int(line.split()[-1])
    pts = np.frombuffer(raw, dtype=np.float32, count=n * 3, offset=end)
    return pts, n


def setup_foam(water_obj, radius):
    fm = bpy.data.materials.get('Foam')
    if fm is None:
        fm = bpy.data.materials.new('Foam')
        fm.use_nodes = True
        bsdf = fm.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.93, 0.96, 1.0, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.85
            try:
                bsdf.inputs['Emission Color'].default_value = (1.0, 1.0, 1.0, 1.0)
                bsdf.inputs['Emission Strength'].default_value = 0.07
            except KeyError:
                pass

    ng = bpy.data.node_groups.new('FoamPointsGN', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    n_in = ng.nodes.new('NodeGroupInput')
    n_out = ng.nodes.new('NodeGroupOutput')
    m2p = ng.nodes.new('GeometryNodeMeshToPoints')
    m2p.mode = 'VERTICES'
    m2p.inputs['Radius'].default_value = radius
    smat = ng.nodes.new('GeometryNodeSetMaterial')
    smat.inputs['Material'].default_value = fm
    ng.links.new(n_in.outputs[0], m2p.inputs['Mesh'])
    ng.links.new(m2p.outputs['Points'], smat.inputs['Geometry'])
    ng.links.new(smat.outputs['Geometry'], n_out.inputs[0])

    mesh = bpy.data.meshes.new('FoamMesh')
    foam = bpy.data.objects.new('FoamPoints', mesh)
    bpy.context.scene.collection.objects.link(foam)
    foam.matrix_world = water_obj.matrix_world.copy()   # misma transform validada
    mod = foam.modifiers.new('foam', 'NODES')
    mod.node_group = ng
    print(f'[04] capa de espuma activa (radio {radius} m sim)')
    return foam


def swap_foam(foam_obj, ply_path):
    old = foam_obj.data
    mesh = bpy.data.meshes.new('FoamMesh')
    if ply_path is not None and Path(ply_path).exists():
        pts, n = read_foam_ply(ply_path)
        mesh.vertices.add(n)
        mesh.vertices.foreach_set('co', pts)
        mesh.update()
    foam_obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return len(mesh.vertices)


def enable_gpu():
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
    except KeyError:
        bpy.ops.preferences.addon_enable(module='cycles')
        prefs = bpy.context.preferences.addons['cycles'].preferences
    for dev_type in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            found = False
            for d in prefs.devices:
                d.use = (d.type == dev_type)
                found = found or d.use
            if found:
                print(f'[04] GPU backend: {dev_type}')
                return
        except Exception as e:
            print(f'[04] {dev_type} no disponible: {e}')
    print('[04] AVISO: render por CPU')


def frame_index(p):
    return int(''.join(c for c in p.stem if c.isdigit()))


def find_water_object(water_mat):
    cands = [o for o in bpy.data.objects
             if o.type == 'MESH'
             and 'flood' not in o.name.lower()
             and any(m and m.name == water_mat.name for m in o.data.materials)]
    if not cands:
        cands = [o for o in bpy.data.objects
                 if o.type == 'MESH' and 'flood' not in o.name.lower()
                 and len(o.data.vertices) > 1000]
    if not cands:
        return None
    return max(cands, key=lambda o: len(o.data.vertices))


def main():
    args = parse_args()
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    enable_gpu()
    scene.cycles.device = 'GPU'
    if args.samples:
        scene.cycles.samples = args.samples
    if args.bounces is not None:
        scene.cycles.max_bounces = args.bounces
        scene.cycles.transmission_bounces = args.bounces
        scene.cycles.transparent_max_bounces = args.bounces
        print(f'[04] bounces -> {args.bounces}')
    if args.camera:
        cam = bpy.data.objects.get(args.camera)
        if cam is None or cam.type != 'CAMERA':
            print(f'[04] ERROR: no existe la camara "{args.camera}" en el .blend'); return
        scene.camera = cam
        print(f'[04] camara -> {args.camera}')

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = Path(args.meshes)

    water_mat = bpy.data.materials.get('Water')
    if water_mat is None:
        print('[04] ERROR: no encuentro el material "Water" en el .blend.'); return

    if args.absorption is not None:
        for n in water_mat.node_tree.nodes:
            if n.type == 'VOLUME_ABSORPTION':
                n.inputs['Density'].default_value = args.absorption
                print(f'[04] Volume Absorption density -> {args.absorption}')

    water_obj = find_water_object(water_mat)
    if water_obj is None:
        print('[04] ERROR: no encuentro el objeto de agua en el .blend.'); return
    print(f'[04] objeto de agua elegido: {water_obj.name}  ({len(water_obj.data.vertices):,} verts) -- la malla se intercambia por frame')

    foam_obj = None
    if args.foam_dir:
        foam_dir = Path(args.foam_dir)
        if np is None:
            print('[04] AVISO: este Blender no trae numpy — render SIN espuma')
        elif not foam_dir.is_dir():
            print(f'[04] AVISO: --foam-dir {foam_dir} no existe — render sin espuma')
        else:
            foam_obj = setup_foam(water_obj, args.foam_radius)

    objs = sorted(mesh_dir.glob('frame_*.obj'), key=frame_index)
    objs = [o for o in objs
            if args.start <= frame_index(o) <= args.end and frame_index(o) % args.stride == 0]
    if not objs:
        print(f'[04] No hay OBJ en {mesh_dir} en el rango pedido.'); return
    print(f'[04] {len(objs)} frames a renderizar')

    for i, obj_path in enumerate(objs):
        idx = frame_index(obj_path)
        out_png = out_dir / f'frame_{idx:06d}.png'
        if out_png.exists():
            continue

        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(obj_path.resolve()),
                              up_axis='Y', forward_axis='NEGATIVE_Z')
        added = [o for o in bpy.data.objects if o not in before]
        tmp = next((o for o in added if o.type == 'MESH'), None)
        if tmp is None:
            print(f'[04] FAIL import frame {idx}'); continue

        old_mesh = water_obj.data
        water_obj.data = tmp.data
        water_obj.data.materials.clear()
        water_obj.data.materials.append(water_mat)
        for p in water_obj.data.polygons:
            p.use_smooth = True

        for o in added:
            bpy.data.objects.remove(o, do_unlink=True)
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

        n_foam = 0
        if foam_obj is not None:
            n_foam = swap_foam(foam_obj, Path(args.foam_dir) / f'foam_{idx:06d}.ply')

        scene.render.filepath = str(out_png)
        print(f'[04] [{i+1}/{len(objs)}] frame {idx} ({len(water_obj.data.vertices):,} verts, foam {n_foam:,}) -> {out_png.name}')
        bpy.ops.render.render(write_still=True)

    print('[04] Secuencia lista.')


main()
