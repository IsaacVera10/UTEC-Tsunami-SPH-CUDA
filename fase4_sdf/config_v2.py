"""
Config v2 — Integración de UTEC como boundary SDF en la simulación SPH.

Derivación de coordenadas (sesión de calibración):
  La transform del agua en escena_utec_v2.blend (RotX=90, RotZ=90,
  Scale=(2.5, 2.0, 3.0), Loc=(-13.421, -98.178, 0)) mapea sim -> Blender como:
      world_X = 3.0*z_sim - 13.421
      world_Y = 2.5*x_sim - 98.178
      world_Z = 2.0*y_sim
  Invirtiendo esa transform y aplicandola al CampusRoot
  (Loc=(30.03, 14.972, 14.562), Scale=35, GLB crudo 1.90 x 0.83 x 1.03):
  UTEC queda en sim-space en el bbox de abajo.

Convencion de ejes en SIM-SPACE (la del solver CUDA):
  X = direccion de avance de la ola (0 -> 60)
  Y = altura
  Z = profundidad/ancho
"""

DOMAIN_MIN = (0.0,  0.0,  0.0)
DOMAIN_MAX = (60.0, 30.0, 30.0)   # Y duplicado: techo a 2x la altura de UTEC

UTEC_BBOX_MIN = (38.0, 0.0,  3.4)    # esquina min del edificio en sim-space
UTEC_BBOX_MAX = (52.5, 14.5, 25.6)   # esquina max

SDF_MARGIN = 2.0

SDF_VOXEL = 0.25     # m/celda. 0.25 recomendado (reporte diagnostico).


GLB_INPUT       = "UTEC_RAW.glb"     # usar RAW, no _texture (fragmentado)
MESH_CLEAN_OUT  = "utec_clean.obj"   # malla watertight resultante
VOXEL_REMESH    = 0.01               # tamano de voxel del remesh EN UNIDADES

H       = 0.5
SPACING = H * 0.27          # 0.135 m
SDF_STIFFNESS = 3000.0      # k de la fuerza de penalizacion del SDF
