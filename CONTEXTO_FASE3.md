# CONTEXTO — Tsunami SPH UTEC, pipeline cine v4 (handoff completo)

**Última actualización:** 2026-07-08
**Estado:** look final bloqueado tras 5 iteraciones de gate. Pipeline v4 validado
con rangos reanudables para mallas y renders.
Una IA/persona nueva debe poder retomar desde aquí sin leer el chat.

---

## 1. Resumen ejecutivo

Simulación WCSPH CUDA de un tsunami golpeando el campus UTEC, con colisión física
real contra el edificio (SDF precomputado) y render Cycles con capa de espuma.

```text
solver CUDA (fase1, SDF boundary)          125 fps efectivos, 4M partículas
  -> cache_cine/frame_*.bin  (1450 frames = 11.6 s físicos)
  -> 07_extract_foam.py      separa BULK (a mallar) y FOAM (spray blanco)
  -> 02_splashsurf.py        marching cubes sobre el bulk (2 recetas)
  -> 04_render_sequence.py   Blender Cycles, swap de malla + point cloud de foam
  -> ffmpeg a 60 fps         => slow-motion 2.08x real
```

**Todo corre en Khipu** (login sin internet; instalaciones offline ya hechas: pysplashsurf
0.14.1 vía wheel + shim `~/bin/splashsurf`, Blender 5.1.2 tarball en `$HOME`).

## 2. Presets FINALES (bloqueados — no iterar más el look)

| Cámara | Malla | Espuma | Absorción | Render args (slurm v4) |
|---|---|---|---|---|
| `Cam_AereaEpic` | continuous | **NINGUNA** | 0.018 | `"" 0.022 0.018` |
| `Cam_AzoteaPOV` | granular | foam_cine, radio 0.018 | default (0.045) | `foam_cine 0.018` |
| `Cam_DetalleImpacto` | granular | foam_cine, radio 0.018 | default | `foam_cine 0.018` |

Veredicto A/B/C (consenso Claude+Codex+usuario, 2026-07-08): en plano aéreo la espuma
como puntos SIEMPRE lee como ruido/puntillismo (probado con 236k, 60k y 30k puntos, radios
0.012-0.03); la masa de agua limpia con contorno de splash es lo cinematográfico. La espuma
solo funciona CERCA de cámara (POV/Detalle), donde el spray vende caos.

**Recetas de superficie (en `khipu_job_cine_splash_range.slurm`):**
```text
continuous: radius=0.115 smooth=3.2 cube=2.5 thr=0.50 iters=30/15  JOBS=1 THREADS=32
granular:   radius=0.105 smooth=2.6 cube=1.0 thr=0.55 iters=15/8   JOBS=2 THREADS=16
foam (07 dentro del slurm): --max-box-neighbors 40 --min-y 1.6 --max-points 400000
```

## 3. Simulación v4 (hecha, job 44975)

- `./sph_tsunami 1450 cache_cine --sdf sdf/utec_sdf.bin --particles 4500000
  --spacing-mult 0.78 --target-fps 125`
- Resultado: N=3,922,371 (tras filtro de spawn), spacing 0.1053 m, 8 pasos/frame
  (125 fps exactos), **0 penetraciones φ<-0.3, 0 NaN** en los 1450 frames.
- Timing físico (medido por curva de contactos): frente golpea t≈1.5 s (frame ~190),
  pico de trepada t≈3.3 s (~420), clímax de acumulación t≈6 s (~760), calma ~1330.
- 1450 frames @125 → a 60 fps de video = 24.2 s de material en slow-mo 2.08×.

## 4. Sistema de espuma (07_extract_foam.py) — cómo funciona

- Clasifica spray por **conteo de vecinos en caja 3×3×3** (celda = 2.2×spacing),
  vectorizado numpy. Umbral en CONTEOS → independiente de la resolución.
- `--min-y 1.6`: solo spray que vuela >0.6 m sobre el nivel del mar (h0=1.0);
  sin esto la capa superficial picada de 4M partículas se clasifica entera.
- `--wall-margin 1.0`: el spray pegado a las paredes del dominio va al bulk.
- **Separación anti "ojo de pez"**: `--bulk-output` escribe el PLY SIN spray y
  splashsurf malla ESO — si no, cada gota aislada se vuelve blob de vidrio con
  esfera blanca adentro. El splash slurm ya hace todo esto internamente (paso 07
  reemplazó al paso 01).
- Render: `04` con `--foam-dir` crea point cloud nativo (GN Mesh-to-Points +
  material Foam blanco, emisión 0.07) alineado copiando la matrix_world del agua.
  Lee los PLY a mano con numpy (el importador de Blender puede rotar los datos).

## 5. Escena v4 (`06_setup_render_scene_v4_cine.py` → `fase3/escena_utec_v4_cine.blend`)

Generada 100% por script desde escena_utec_v2.blend (que NUNCA se toca — tiene la
transform validada del agua: loc(-13.421,-98.178,0) RotXZ 90° scale(2.5,2.0,3.0)).
Contiene, en orden de descubrimiento:
- 1920×1080, AgX Medium High Contrast, exposure +0.35, Cycles 128 samples.
- **Seabed**: plano arena 4000 m en z=-0.05 — SIN él no hay fondo y la losa se ve
  pálida (luz al cielo) mientras el dominio cerrado absorbe (negro) = look "isla".
- **FloodSlab tope z=2.1** (10 cm sobre el nivel del mar): tapa la costura del borde
  del dominio y el agua calmada; solo la ola/splash asoma.
- Oleaje procedural en el material Water (bump por posición MUNDO — consistente
  entre losa infinita y agua sim).
- Cámaras: `Cam_AereaEpic` (-60,-136,80)→(31,2,8) f37 — el dominio debe LLENAR el
  cuadro; más lejos = "sello postal flotante". `Cam_AzoteaPOV` (25,6,24.1)→(25,-72,7)
  f24 (parado en el techo real, z=22.1 + 2). `Cam_DetalleImpacto` (-24,-62,52)→(38,2,11) f42.
- Lecciones de cámara: el modelo Meshy NO tiene interiores (cámara dentro = negro);
  cámaras rasantes miran a través de decenas de metros de spray (frame lechoso);
  siempre FUERA del dominio de agua (mundo X∈[-13,77] Y∈[-98,52]).

## 6. Límites del cluster

```text
a-tesis:    cpu=32, mem=98G, walltime=10h
a-pregrado: cpu=32, mem=98G, gpu=1, walltime=8h
```
- sbatch con --mem>98G queda PD eterno con "QOSMaxMemoryPerUser".
- Enviar >3 jobs a la vez: los excedentes se rechazan → lanzar en tandas de ≤3.
- Los límites se aplican por QOS. Dos rangos pueden correr en paralelo cuando cada
  uno usa un QOS distinto.
- **Memoria de splashsurf escala con SMOOTH, no con CUBE** (evidencia: cube 2.0→2.5
  no movió MaxRSS; smooth 4.0 = OOM garantizado en frames densos; 3.2 cabe).
- **2 splashsurf continuous simultáneos se ahogan** (job 45108: MaxRSS 90G/96G,
  7.6/32 cores efectivos, 0 OBJs en 2.5 h) → continuous corre JOBS=1×32 hilos.
- OOM del cgroup mata SILENCIOSO (.err vacío, exit en segundos). Diagnóstico:
  `sacct -j <id> --format=JobID,State,ReqMem,MaxRSS,Elapsed`.

## 7. Batch final

```bash
# Mallas CPU en QOS separados:
sbatch --qos=a-tesis --time=10:00:00 scripts/khipu_job_cine_splash_range.slurm continuous 0 724
sbatch --qos=a-pregrado --time=08:00:00 scripts/khipu_job_cine_splash_range.slurm continuous 725 1449
sbatch --qos=a-pregrado --time=08:00:00 scripts/khipu_job_cine_splash_range.slurm granular 300 800

# Renders GPU tras el conteo completo de OBJ:
sbatch scripts/khipu_job_render_v4.slurm Cam_AereaEpic mesh_cine_continuous_0_724   0   724  1 "" 0.022 0.018
sbatch scripts/khipu_job_render_v4.slurm Cam_AereaEpic mesh_cine_continuous_725_1449 725 1449 1 "" 0.022 0.018
sbatch scripts/khipu_job_render_v4.slurm Cam_AzoteaPOV mesh_cine_granular_300_800   300 800  1 foam_cine 0.018

# TANDA 3:
sbatch scripts/khipu_job_render_v4.slurm Cam_DetalleImpacto mesh_cine_granular_300_800 300 800 1 foam_cine 0.018
```

Args del render slurm: `<Cam> <mesh_dir> <start> <end> <stride> [foam_dir] [foam_radius] [absorption] [out_suffix]`.
Todo es reanudable: mallas y PNGs existentes se saltan; si un job muere por walltime,
re-sbatch idéntico. Los PLY del rango se reutilizan si existen (`ply_cine_<tag>/`).

```bash
# FINAL (local, tras bajar render_v4_Cam_* por WinSCP):
ffmpeg -framerate 60 -start_number 0   -i render_v4_Cam_AereaEpic/frame_%06d.png    -c:v libx264 -pix_fmt yuv420p -crf 17 -movflags +faststart tsunami_aerea.mp4
ffmpeg -framerate 60 -start_number 300 -i render_v4_Cam_AzoteaPOV/frame_%06d.png    -c:v libx264 -pix_fmt yuv420p -crf 17 -movflags +faststart tsunami_azotea.mp4
ffmpeg -framerate 60 -start_number 300 -i render_v4_Cam_DetalleImpacto/frame_%06d.png -c:v libx264 -pix_fmt yuv420p -crf 17 -movflags +faststart tsunami_detalle.mp4
```
Edición sugerida: aérea como plano de escala (24 s), cortes de azotea/detalle (8 s c/u)
en el impacto. La espuma fuerte vive en los cortes cercanos; el plano general va limpio.

## 8. Coordenadas y convenciones (NO romper)

```text
sim:    X=avance ola (0-60), Y=altura (0-30), Z=profundidad (0-30)
mundo:  world_X = 3.0*z_sim - 13.421   |  world_Y = 2.5*x_sim - 98.178
        world_Z = 2.0*y_sim            |  océano en Y=-98, ola avanza +Y
edificio (SDF): sim (38,0,3.4)-(52.5,14.5,25.6); techo ppal mundo z≈22
```
`04_render_sequence.py` intercambia SOLO geometría y conserva la transform del agua.
NUNCA Ctrl+A ni tocar rotaciones del objeto de agua. El SDF: orden x-lento/z-rápido,
φ<0 dentro, swap de ejes solo en `fetchSDF()` (tex3D recibe z,y,x).

## 9. Deploy y archivos

`bash fase1_sph_tsunami/scripts/deploy_khipu_cine.sh` sube: main.cu, compile_login.sh,
SLURMs v4, 01/02/04/06/07 de fase2, y el blend v4 si existe (regenerarlo localmente:
`blender -b fase3/escena_utec_v2.blend -P fase2_splashsurf/06_setup_render_scene_v4_cine.py
-- <RUTA ABSOLUTA salida>` — con ruta relativa Blender lo escribe en C:\). El deploy
normaliza CRLF en remoto (git local tiene autocrlf=true; .gitattributes protege *.sh/*.slurm).

## 10. Docs hermanos

- `fase1_sph_tsunami/INFORME_TECNICO.md` — física y CUDA (SPH + SDF), para estudiar.
- `fase2_splashsurf/LOGICA.md` — el porqué de cada decisión (incluye v4/espuma).
- `fase2_splashsurf/EJECUCION.md` — manual operativo del pipeline v4.
- `fase4_sdf/README_V2_SDF.md` — bitácora de la integración SDF (etapas 0-3).
