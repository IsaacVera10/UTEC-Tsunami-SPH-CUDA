
import bpy
import math
import mathutils
import sys

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT_BLEND = argv[0] if argv else "escena_utec_v4_cine.blend"

scene = bpy.context.scene

scene.render.engine = 'CYCLES'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.cycles.samples = 128
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 8
scene.cycles.transmission_bounces = 8
scene.cycles.transparent_max_bounces = 8

try:
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.35
    scene.view_settings.gamma = 1.0
    print('[cine] color: AgX Medium High Contrast, exposure +0.35')
except Exception as e:
    print(f'[cine] AgX no disponible ({e})')

slab = bpy.data.objects.get('FloodSlab')
if slab:
    slab.scale.z = 2.1
    slab.location.z = 1.05
    print('[cine] FloodSlab tope -> z=2.1 m (cubre costura del dominio)')

seabed = bpy.data.objects.get('Seabed')
if seabed is None:
    bpy.ops.mesh.primitive_plane_add(size=4000.0, location=(30.0, 15.0, -0.05))
    seabed = bpy.context.active_object
    seabed.name = 'Seabed'
    sm = bpy.data.materials.new('Sand')
    sm.use_nodes = True
    sb = sm.node_tree.nodes.get('Principled BSDF')
    if sb:
        sb.inputs['Base Color'].default_value = (0.23, 0.20, 0.16, 1.0)
        sb.inputs['Roughness'].default_value = 0.95
    seabed.data.materials.append(sm)
    print('[cine] Seabed: plano arena 4000 m en z=-0.05')

water = bpy.data.materials.get('Water')
if water and water.use_nodes:
    wt = water.node_tree
    glass = None
    for n in wt.nodes:
        if n.type == 'BSDF_GLASS':
            glass = n
            n.inputs['Roughness'].default_value = 0.08
            n.inputs['Color'].default_value = (0.72, 0.86, 0.90, 1.0)
        elif n.type == 'VOLUME_ABSORPTION':
            n.inputs['Color'].default_value = (0.12, 0.38, 0.45, 1.0)
            n.inputs['Density'].default_value = 0.045
    print('[cine] Water: roughness 0.08, absorcion turquesa densa')

    if glass is not None:
        geo = wt.nodes.new('ShaderNodeNewGeometry')
        noise = wt.nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 0.08      # olas de ~12 m
        noise.inputs['Detail'].default_value = 6.0
        noise.inputs['Roughness'].default_value = 0.55
        bump = wt.nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = 0.28
        bump.inputs['Distance'].default_value = 0.35
        wt.links.new(geo.outputs['Position'], noise.inputs['Vector'])
        wt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
        wt.links.new(bump.outputs['Normal'], glass.inputs['Normal'])
        print('[cine] Water: oleaje procedural (bump por posicion mundo)')

sun_dir = mathutils.Vector((-0.34, -0.58, 0.74)).normalized()
world = scene.world
world.use_nodes = True
wnt = world.node_tree
wnt.nodes.clear()
wout = wnt.nodes.new('ShaderNodeOutputWorld')
bg = wnt.nodes.new('ShaderNodeBackground')
sky = wnt.nodes.new('ShaderNodeTexSky')
sky.sky_type = 'HOSEK_WILKIE'
sky.sun_direction = sun_dir
sky.turbidity = 3.0
sky.ground_albedo = 0.35
bg.inputs['Strength'].default_value = 1.9
wnt.links.new(sky.outputs[0], bg.inputs['Color'])
wnt.links.new(bg.outputs['Background'], wout.inputs['Surface'])
print(f'[cine] cielo Hosek, sol dir {tuple(round(v,2) for v in sun_dir)}')

sun = bpy.data.objects.get('Sun')
if sun is None:
    sd = bpy.data.lights.new('Sun', 'SUN')
    sun = bpy.data.objects.new('Sun', sd)
    scene.collection.objects.link(sun)
sun.rotation_euler = (-sun_dir).to_track_quat('-Z', 'Y').to_euler()
sun.data.energy = 5.5
sun.data.angle = math.radians(0.8)
sun.data.color = (1.0, 0.88, 0.72)
print('[cine] Sun: 5.5 W/m2, calido')


def make_cam(name, loc, target, lens):
    cam = bpy.data.objects.get(name)
    if cam is None:
        cd = bpy.data.cameras.new(name)
        cam = bpy.data.objects.new(name, cd)
        scene.collection.objects.link(cam)
    cam.location = mathutils.Vector(loc)
    d = mathutils.Vector(target) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = lens
    cam.data.clip_start = 0.1
    cam.data.clip_end = 1200.0
    print(f'[cine] {name}: loc={loc} -> {target}, lente {lens}')
    return cam


aerea = make_cam('Cam_AereaEpic',
                 (-60.0, -136.0, 80.0),
                 (31.0, 2.0, 8.0),
                 37)

azotea = make_cam('Cam_AzoteaPOV',
                  (25.0, 6.0, 24.1),
                  (25.0, -72.0, 7.0),
                  24)

detalle = make_cam('Cam_DetalleImpacto',
                   (-24.0, -62.0, 52.0),
                   (38.0, 2.0, 11.0),
                   42)

scene.camera = aerea

bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)
print(f'[cine] guardado -> {OUT_BLEND}')
