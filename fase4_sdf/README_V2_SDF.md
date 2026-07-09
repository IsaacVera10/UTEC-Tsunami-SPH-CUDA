# Fase 3-SDF — Interacción física agua ↔ UTEC (v2)

> Registro histórico de calibración. La entrega final usa
> `khipu_job_cine_sim.slurm`; los nombres `khipu_job_v2.slurm` y
> `khipu_job_v3.slurm` corresponden a corridas anteriores.

Integración del edificio UTEC como boundary condition (SDF) en el solver SPH,
para que las partículas choquen/rodeen la estructura en vez de atravesarla.

## Coordenadas amarradas (NO recalcular — derivadas de escena_utec_v2.blend)

| Cosa | Valor |
|---|---|
| Dominio v2 | (0,0,0) → (60, **30**, 30) — techo duplicado vs v1 |
| UTEC bbox en sim-space | (38.0, 0.0, 3.4) → (52.5, 14.5, 25.6) |
| Pista de la ola | ~38 m desde X=0 hasta la fachada |
| Objetivo visual | agua trepa a ~10-12 m en el impacto (penúltimos pisos) |

Derivación: transform del agua en Blender (RotX=90, RotZ=90, Scale 2.5/2.0/3.0,
Loc -13.421/-98.178/0) invertida y aplicada al CampusRoot (Loc 30.03/14.972/14.562,
Scale 35). La composición visual validada ES el mapeo físico.

## Etapas (correr en orden, no saltarse validaciones)

### Etapa 0 — SDF sin simular  [LOCAL, sin GPU] — ✅ COMPLETADA
```
pip install trimesh open3d numpy scikit-image
python 01_clean_mesh.py          # UTEC_RAW.glb -> utec_clean.obj (watertight)
python 02_generate_sdf.py        # -> sdf/utec_sdf.bin + meta + check.obj
```
**Validación (aprobada):** `sdf/utec_sdf_check.obj` coincide en forma con el
GLB original en Blender. SDF resultante: 75×75×106, voxel 0.25 m, sin fugas.

### Etapa 1 — Mini-sim local  [RTX 4060] — ✅ COMPLETADA (2026-07-02)
Solver modificado: `--sdf <path>` carga el .bin a textura 3D CUDA (swap de
ejes documentado en `fetchSDF()` de kernels.cu), fuerza de penalización
`k·(−φ)·n̂` con gradiente por diferencias centrales + amortiguamiento de
velocidad normal en la integración + filtro de spawn dentro del footprint.
- **Test A (orientación):** φ = −0.057 en el centro del edificio, positivo/
  fuera de grilla en los otros 3 puntos de control. ✅
- **Test B (mini-sim 300 frames, 153k part.):** 0 penetraciones φ<−0.3,
  0 NaN/Inf, agua apilada contra la fachada y rodeando por los corredores. ✅
- **Test C (visual, splashsurf + MeshLab vs utec_sdf_check.obj):** molde
  limpio del edificio, splash sobre azotea, seco detrás, cero invasión. ✅

### Etapa 2 — Calibración en Khipu  [A100, sm_80] — 🔄 EN CURSO (2026-07-03)
300 frames a resolución completa con SDF activo (`scripts/khipu_job_v2.slurm`),
output a `cache_sdf_calib/`. **Objetivo de calibración: el agua debe trepar a
~10-12 m contra la fachada** (el edificio llega a y=14.5).

Deploy desde local (sube SOLO los archivos que cambiaron vs v1 — no builds
ni caches): `bash fase1_sph_tsunami/scripts/deploy_khipu.sh`, y luego:
```
ssh usuario@khipu.utec.edu.pe
cd ~/computer_graphics/fase1_sph_tsunami
bash scripts/compile_login.sh        # login node, sm_80, igual que v1
sbatch scripts/khipu_job_v2.slurm
squeue -u "$USER"
```
Chequear en `logs/sph_sdf_<jobid>.out` apenas arranque:
- Carga del SDF: `dims 75x75x106, voxel 0.250 m, origin (36.00, -2.00, 1.40)`
  y el Test A impreso con `p0 ... phi = -0.057 (DENTRO)`.
- Init N-wave: **1,286,656 partículas → 1,250,504 tras el filtro** del
  footprint (−36,152). OJO: a resolución completa el perfil N-wave llena
  ~1.29M, NO los ~1.8M del comentario del SLURM (ese número era del init
  antiguo tipo dam-break). Verificado en el smoke test local (2026-07-03).
- `[sdf] frame N: ... profundos(phi<-0.3)=0` en todos los frames (como en
  la mini-sim). Si aparece `<-- TUNNELING`, subir `sdf_k` (make_params).

**Knobs de calibración** (en `init_tsunami_wave()` de
`fase1_sph_tsunami/src/main.cu`) si el agua no llega a 10-12 m:
1. `A_crest` — cap actual `fminf(9.f, ...)`: subir a 11-12 para una cresta
   más alta.
2. `h0 = 0.5f` — lámina en reposo: subir a 1.0-1.5 para más masa de impacto.

Cada iteración = editar main.cu + `bash scripts/compile_login.sh` (~30 s en
el login node) + re-sbatch. Evaluar con el maxY del log y 2-3 frames
reconstruidos localmente (fase2_splashsurf 01+02, MeshLab con
`sdf/utec_sdf_check.obj`).

**Bitácora de calibración:**
- **Iter 1** (job 43870, A_crest=9, h0=0.5, N=1,250,512): 0 tunneling y 0 NaN
  en 300 frames, pero el agua solo trepa a **p95≈4.8 m / p99≈8.1 m** en la
  fachada (pico ~frame 200; medido en frames 150/200/250/299, franja
  x∈[37.5,39.5]). Corto vs objetivo 10-12 m — la ola se disipa cruzando 30 m
  de lámina de 0.5 m.
- **Iter 2** (job 44277, A_crest=12, h0=1.0, N=1,866,488): 0 tunneling y
  0 NaN en 300 frames (corrió en RTX A6000 — el binario sm_80 es compatible).
  **✅ OBJETIVO ALCANZADO en el pico (frame ~200):** altura mojada por metro
  de fachada (p99 de Y en slab x∈[37.8,39.0], bins de 1 m en z) da
  **11–13.5 m en el tercio z=3.4–11.4** (mediana global 9.0 m, 8/23 bins
  ≥10 m, max 13.5 m) — lengua coherente, no spray. En frame 250 ya recede
  (mediana 8.1 m). OJO con la métrica: el p95 agregado de la franja da solo
  6.4 m porque la piscina profunda al pie arrastra el percentil — usar el
  perfil por bins, no el agregado.
- **Confirmación visual (2026-07-04): ✅ APROBADA.** Frames 200/250/299
  reconstruidos con splashsurf y superpuestos con utec_sdf_check.obj en
  MeshLab: la ola golpea y envuelve el edificio sin atravesarlo; el caos
  post-impacto de los frames tardíos es el chapoteo esperado de la caja
  cerrada (igual que v1). **CALIBRACIÓN CERRADA: A_crest=12, h0=1.0.**
  Nota: utec_clean.obj NO alinea en sim-space (está en unidades del GLB
  crudo, ~1.9 u.) — para superposiciones usar siempre utec_sdf_check.obj;
  el render final usa el modelo real de escena_utec_v2.blend.

### Etapa 3 — Corrida completa — ✅ SIMULACIÓN COMPLETADA (2026-07-04)
Job 44333 (RTX A6000): 900 frames, N=1,866,488, 0 tunneling, 0 NaN.
Contactos pico ~34.8k en frames 390-410. Momentos clave: impacto frame
~120-200 (pico visual ~200), clímax ~400, calma hacia 900. Falta el
post-proceso local (abajo).

900 frames (15.3 s sim) con la calibración congelada, vía
`fase1_sph_tsunami/scripts/khipu_job_v3.slurm` → output `cache_final/` +
`cache_final.tar.gz` (~10 GB comprimido, ~20 GB sin comprimir — verificar
cuota en Khipu). Lanzamiento:
```
bash fase1_sph_tsunami/scripts/deploy_khipu.sh   # sube el v3 (ya incluido)
ssh usuario@khipu.utec.edu.pe
cd ~/computer_graphics/fase1_sph_tsunami
sbatch scripts/khipu_job_v3.slurm    # el binario ya está compilado de la iter 2
```
Chequeos del log (`logs/sph_final_<jobid>.out`): init 1,908,492 → 1,866,48x,
`profundos(phi<-0.3)=0` sostenido. Luego: bajar el tar → pipeline
fase2_splashsurf (01+02) → render con las DOS cámaras (aérea + detalle del
impacto; el pico del impacto vive ~frame 200, t≈3.4 s).

## SLURM v2

El job real vive en `fase1_sph_tsunami/scripts/khipu_job_v2.slurm` (300
frames, A100, 04:00:00, output a `cache_sdf_calib/` + tar.gz). Asume que el
sbatch se hace DESDE `~/computer_graphics/fase1_sph_tsunami/` (usa
`$SLURM_SUBMIT_DIR`) y que el SDF está en `sdf/utec_sdf.bin` relativo a esa
carpeta — `deploy_khipu.sh` deja todo en su sitio, incluido el `logs/` que
`#SBATCH --output` necesita ANTES del submit.

## Dos cámaras para el render (pedido del profesor)

En `escena_utec_v2.blend` agregar `Cam_Detalle` a nivel de calle mirando el
punto de impacto de la fachada. Al `04_render_sequence.py` se le agrega
`--camera <nombre>` (cambio de ~5 líneas) y se corren dos batches:
```
... --out D:\render_aerea    --camera Cam
... --out D:\render_detalle  --camera Cam_Detalle --start 100 --end 400
```
La toma detalle solo necesita el rango del impacto, no los 900 frames.

## Qué NO subir a GitHub
`UTEC_RAW.glb`, `utec_clean.obj`, `sdf/*.bin` (pesados). Sí subir: los .py,
este README, y `sdf/utec_sdf_meta.json`.
