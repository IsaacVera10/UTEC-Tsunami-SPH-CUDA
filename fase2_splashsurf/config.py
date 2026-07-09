from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

H       = 0.5
SPACING = H * 0.27

DOMAIN_MIN = (0.0,  0.0,  0.0)
DOMAIN_MAX = (60.0, 30.0, 30.0)

SURF_RADIUS     = SPACING
SURF_SMOOTH = 3.0   # multiplicador real del kernel (post-fix de unidades en 02).
SURF_CUBE_SIZE  = 1.5   # subido de 1.5 -> evita OOM en frames densos del inicio
SURF_THRESHOLD  = 0.6
SURF_SMOOTH_ITERS = 25
SURF_NORMAL_ITERS = 10

BIN_DIR  = ROOT / "sph_tsunami" / "output"
PLY_DIR  = ROOT / "pipeline"   / "ply"
MESH_DIR = ROOT / "pipeline"   / "meshes"
ABC_PATH = ROOT / "pipeline"   / "tsunami.abc"
