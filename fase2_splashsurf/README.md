# Fase 2: superficie, espuma y render

## Propósito

Esta fase separa el agua continua del spray, genera una superficie triangular y
reemplaza la malla de agua en Blender para cada frame.

## Homologación técnica

| Etapa | Archivo | Resultado |
|---|---|---|
| Separación bulk/foam | `07_extract_foam.py` | PLY binarios |
| Reconstrucción | `02_splashsurf.py` | OBJ |
| Configuración visual | `06_setup_render_scene_v4_cine.py` | Blend v4 |
| Secuencia Cycles | `04_render_sequence.py` | PNG |
| Parámetros base | `config.py` | constantes compartidas |

## Clasificación de espuma

`07_extract_foam.py` divide el espacio en celdas y suma partículas en las 27 celdas
vecinas. Una partícula entra en la espuma cuando cumple:

```text
neighbor_count <= 40
y > 1.6 m
inside_domain_margin
```

La máscara complementaria forma el bulk. El bulk entra en SplashSurf; la espuma
entra en Geometry Nodes como puntos blancos.

## Recetas finales

| Modo | Radio | Smooth | Cube | Threshold | Iteraciones |
|---|---:|---:|---:|---:|---:|
| `continuous` | 0.115 | 3.2 | 2.5 | 0.50 | 30/15 |
| `granular` | 0.105 | 2.6 | 1.0 | 0.55 | 15/8 |

La cámara aérea usa `continuous` sin espuma. Las cámaras de azotea e impacto usan
`granular` con espuma de radio `0.018`.

## Ejecución

Primera y segunda mitad aérea:

```bash
sbatch scripts/khipu_job_cine_splash_range.slurm continuous 0 724
sbatch scripts/khipu_job_cine_splash_range.slurm continuous 725 1449
```

Rango cercano:

```bash
sbatch scripts/khipu_job_cine_splash_range.slurm granular 300 800
```

El script conserva OBJ existentes. Un segundo envío continúa desde el primer frame
faltante.

Render aéreo:

```bash
sbatch scripts/khipu_job_render_v4.slurm \
  Cam_AereaEpic mesh_cine_continuous_0_724 \
  0 724 1 "" 0.022 0.018
```

Render de azotea:

```bash
sbatch scripts/khipu_job_render_v4.slurm \
  Cam_AzoteaPOV mesh_cine_granular_300_800 \
  300 800 1 foam_cine 0.018
```

`EJECUCION.md` contiene el manual completo. `LOGICA.md` registra las pruebas A/B/C,
los límites de memoria y las decisiones visuales.
