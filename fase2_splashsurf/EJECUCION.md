# EJECUCIÓN.md — Manual operativo del pipeline v4 CINE

Proyecto: **Simulación de tsunami sobre el campus UTEC** · Computer Graphics, UTEC
Guía de comandos. Para el *porqué* de cada decisión, ver `LOGICA.md`; para la física,
`fase1_sph_tsunami/INFORME_TECNICO.md`; para el estado/handoff, `CONTEXTO_FASE3.md`.

> **El pipeline v4 corre COMPLETO en Khipu** (sim + mallas + separación de spray + render). Localmente
> solo se genera el `.blend` y se hacen previews de 1 frame.

---

## 1. Requisitos

### En Khipu (ya instalado — cómo se replica)
| Qué | Cómo (el login node NO tiene internet) |
|---|---|
| Binario `sph_tsunami` | `bash scripts/compile_login.sh` (login node, sm_80) |
| splashsurf 0.14.1 | wheel `pysplashsurf` de `khipu_wheels/` local → `pip install --user --no-index --find-links ~/computer_graphics/wheels pysplashsurf` (con `module load gnu12/12.4.0 python3/3.11.11`) + shim: `printf '#!/bin/bash\nexec pysplashsurf "$@"\n' > ~/bin/splashsurf && chmod +x ~/bin/splashsurf` |
| Blender 5.1.2 | tarball de `khipu_wheels/` → `tar -xf blender-5.1.2-linux-x64.tar.xz -C ~` (trae numpy) |
| SDF del edificio | `sdf/utec_sdf.bin` + `_meta.json` (subidos por deploy) |

### Local (Windows)
Blender 5.1.2, WinSCP (transferencias con resume), Git Bash (deploy/ffmpeg), ffmpeg.
El repo tiene `.gitattributes` (LF en `*.sh/*.slurm`) — **no** editar scripts de cluster
con herramientas que impongan CRLF.

### Límites del cluster
```text
a-tesis:    cpu=32, mem=98G, walltime=10h
a-pregrado: cpu=32, mem=98G, gpu=1, walltime=8h
```
- `--mem` > 98G ⇒ el job queda `PD (QOSMaxMemoryPerUser)` para siempre.
- Cada QOS admite 32 CPU. Dos jobs de 32 CPU corren juntos cuando usan QOS distintos.
- SplashSurf usa CPU; Blender y WCSPH usan GPU.

---

## 2. Deploy (local → Khipu)

```bash
# 1) regenerar el blend v4 si se tocó 06 (RUTA ABSOLUTA de salida, obligatorio):
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" -b fase3/escena_utec_v2.blend `
  -P fase2_splashsurf/06_setup_render_scene_v4_cine.py -- `
  "D:\ruta\del\repositorio\fase3\escena_utec_v4_cine.blend"

# 2) sube el pipeline (Git Bash):
bash fase1_sph_tsunami/scripts/deploy_khipu_cine.sh \
  usuario@khipu.utec.edu.pe
```
Sube el solver completo, el SDF, los SLURM v4 y los scripts 02/04/06/07. También
sube el blend v4 cuando existe en `fase3/`. Normaliza CRLF en remoto.

---

## 3. Pipeline v4 en Khipu — paso a paso

Todo desde `~/computer_graphics/fase1_sph_tsunami`.

### Paso 1 — Simulación (una vez): 1450 frames @125 fps, ~4M partículas

```bash
bash scripts/compile_login.sh        # si main.cu cambió
sbatch scripts/khipu_job_cine_sim.slurm
```
El job corre: `sph_tsunami 1450 cache_cine --sdf sdf/utec_sdf.bin --particles 4500000
--spacing-mult 0.78 --target-fps 125`. Verificar en `logs/sph_cine_*.out`:
init `4,022,960 → 3,922,371` tras filtro, sanity SDF `p0 ... phi=-0.057 (DENTRO)`,
y `profundos(phi<-0.3)=0` en TODOS los frames (si aparece `<-- TUNNELING`, subir `sdf_k`).
Cache: ~78 GB (revisar `df -h` antes). ~20-25 min en A100.

### Paso 2 — Mallas + espuma (khipu_job_cine_splash_range.slurm)

El job hace TODO por rango: enlaza los `.bin`, corre el **07** (separa bulk/spray:
PLYs sin spray para mallar + `foam_cine/foam_*.ply`), corre el **02** (splashsurf)
en paralelo, pasada secuencial de rescate, y borra los PLY temporales.

```bash
# Primera mitad y segunda mitad en QOS separados:
sbatch --qos=a-tesis --time=10:00:00 scripts/khipu_job_cine_splash_range.slurm continuous 0 724
sbatch --qos=a-pregrado --time=08:00:00 scripts/khipu_job_cine_splash_range.slurm continuous 725 1449

# Rango cercano cuando un QOS quede libre:
sbatch --qos=a-pregrado --time=08:00:00 scripts/khipu_job_cine_splash_range.slurm granular 300 800
```

| Receta | Para qué | Parámetros | Concurrencia |
|---|---|---|---|
| `continuous` | aérea / agua como masa | r=0.115 smooth=3.2 cube=2.5 thr=0.50 | **JOBS=1×32 hilos** (2 simultáneos = thrash de memoria, ver LOGICA §8.6) |
| `granular` | POV / detalle cercano | r=0.105 smooth=2.6 cube=1.0 thr=0.55 | JOBS=2×16 |

Salidas: `mesh_cine_<receta>_<start>_<end>/frame_*.obj` + `foam_cine/` (compartida).
**Reanudable**: mallas y foam existentes se saltan; si muere por walltime, re-sbatch igual.
Verificar al final: `OBJs cine generados: N / N`.

### Paso 3 — Render (khipu_job_render_v4.slurm)

```bash
# uso: sbatch scripts/khipu_job_render_v4.slurm \
#        <Camara> <mesh_dir> <start> <end> <stride> [foam_dir] [foam_radius] [absorption] [out_suffix]
```

**Presets FINALES por cámara (bloqueados tras el test A/B/C):**

```bash
# Aérea: SIN espuma, agua clara
sbatch --qos=a-tesis --time=10:00:00 scripts/khipu_job_render_v4.slurm Cam_AereaEpic mesh_cine_continuous_0_724 0 724 1 "" 0.022 0.018
sbatch --qos=a-pregrado --time=08:00:00 scripts/khipu_job_render_v4.slurm Cam_AereaEpic mesh_cine_continuous_725_1449 725 1449 1 "" 0.022 0.018
# POV azotea y Detalle: granular SIN espuma visible
sbatch scripts/khipu_job_render_v4.slurm Cam_AzoteaPOV      mesh_cine_granular_300_800 300 800 1 "" 0.018
sbatch scripts/khipu_job_render_v4.slurm Cam_DetalleImpacto mesh_cine_granular_300_800 300 800 1 "" 0.018
```

Salida: `render_v4_<Camara><suffix>/frame_*.png` (1080p, 128 samples). Reanudable.
~35-45 s/frame ⇒ la aérea completa son ~15-18 h repartidas en sus 2 jobs.

### Paso 4 — Descargar y ensamblar (local)

Bajar `render_v4_Cam_*` por WinSCP (~2-3 GB total) y:

```bash
ffmpeg -framerate 60 -start_number 0   -i render_v4_Cam_AereaEpic/frame_%06d.png    -c:v libx264 -pix_fmt yuv420p -crf 17 -movflags +faststart tsunami_aerea.mp4
ffmpeg -framerate 60 -start_number 300 -i render_v4_Cam_AzoteaPOV/frame_%06d.png    -c:v libx264 -pix_fmt yuv420p -crf 17 -movflags +faststart tsunami_azotea.mp4
ffmpeg -framerate 60 -start_number 300 -i render_v4_Cam_DetalleImpacto/frame_%06d.png -c:v libx264 -pix_fmt yuv420p -crf 17 -movflags +faststart tsunami_detalle.mp4
```
(Cache a 125 fps montado a 60 = slow-motion 2.08× real.) Edición: aérea como plano
de escala, cortes cercanos para el impacto.

---

## 4. Herramientas auxiliares

### Preview local de un frame
```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" -b fase3\escena_utec_v4_cine.blend `
  -P fase2_splashsurf\04_render_sequence.py -- `
  --meshes <DIRECTORIO_OBJ> --out <DIRECTORIO_PNG> `
  --camera Cam_AereaEpic --start 420 --end 420 --samples 64
```

### 07_extract_foam.py suelto (calibrar espuma)
```bash
python3 ../fase2_splashsurf/07_extract_foam.py --input $PWD/cache_cine --output $PWD/foam_test \
    --spacing 0.105 --max-box-neighbors 40 --min-y 1.6 --max-points 400000 \
    --start 420 --end 420 [--bulk-output DIR] [--jobs 8]
```
Corre en login node (segundos/frame). Los PLY son binarios: inspeccionar con
`grep -a "element vertex" foam_test/foam_000420.ply`.

### Gate de look (OBLIGATORIO antes de cualquier batch)
Renderizar 4-5 frames representativos (aprox/impacto/clímax/calma: ~150/420/760/1330)
con el pipeline completo y revisarlos a ojo. Cada bug de look se caza aquí en minutos.

---

## 5. Troubleshooting

| Síntoma | Causa / arreglo |
|---|---|
| Job `PD (QOSMaxMemoryPerUser)` eterno | `--mem` > 98G. Usa 90G. |
| Job `PD (QOSMaxWallDurationPerJobLimit)` | Usa 10h en `a-tesis` u 8h en `a-pregrado`. |
| Job muere en segundos, `.err` VACÍO | OOM del cgroup (SIGKILL silencioso). `sacct -j <id> --format=JobID,State,ReqMem,MaxRSS,Elapsed`. |
| splashsurf lentísimo, 0 OBJs por horas | 2 procesos continuous simultáneos saturando memoria (MaxRSS≈límite, CPU ~20%). JOBS=1×32 para continuous. |
| OOM en splash aun serial | La memoria escala con SMOOTH, no con CUBE. Bajar smooth (4.0→3.2 validado). |
| `$'\r': command not found` en Khipu | CRLF. `sed -i 's/\r$//' <archivo>` (el deploy lo hace solo). |
| Render: esfera blanca dentro de gota de vidrio | Splashsurf malló el spray. Falta la separación bulk/spray del 07 (`--bulk-output`). |
| Espuma = esferas/estática | Omite `--foam-dir`. El render final no dibuja la capa de puntos. |
| Agua negra vista desde arriba | Absorción 0.045 sobre 10+ m. Override `--absorption 0.018` en esa cámara. |
| Mar circundante pálido vs dominio oscuro ("isla") | Falta el Seabed (el 06 lo crea) o losa por debajo del nivel (tope 2.1). |
| Blender escribe en `C:\...` inesperado | Ruta de salida relativa. SIEMPRE absolutas con Blender headless. |
| `unrecognized arguments` en 02/07 | Script remoto desactualizado. Re-subir el `.py` local. |
| PNG viejos "no se re-renderizan" | Resume salta existentes. Borrar/renombrar el dir de salida o usar `out_suffix`. |
| OBJ casi vacío (1-7 KB) | Header PLY malo o `--radius` sin ajustar al spacing (v4: 0.105-0.115, no 0.135). |

---

## 6. Archivos del pipeline (fase2_splashsurf/)

| Archivo | Rol |
|---|---|
| `config.py` | defaults v2 (los SLURM v4 pasan todo por flags) |
| `02_splashsurf.py` | `.ply` → `.obj`; flags `--radius --cube-size --smooth --threshold --smooth-iters --normal-iters --jobs --threads-per-job` |
| `04_render_sequence.py` | batch Cycles: swap de malla + `--camera --foam-dir --foam-radius --absorption --start/--end/--stride` |
| `06_setup_render_scene_v4_cine.py` | genera `escena_utec_v4_cine.blend` (cámaras, luz, seabed, losa, oleaje) |
| `07_extract_foam.py` | separa bulk/spray; espuma decimada + PLY bulk para splashsurf |
| SLURMs (en `fase1_sph_tsunami/scripts/`) | `khipu_job_cine_sim` / `_cine_splash_range` / `_render_v4` / `deploy_khipu_cine.sh` |
