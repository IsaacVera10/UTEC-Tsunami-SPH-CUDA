# Fase 1: simulación WCSPH en CUDA

## Propósito

Esta fase calcula el movimiento del agua, aplica las paredes del dominio y evalúa
la colisión contra el campus mediante una textura SDF 3D.

## Homologación técnica

| Concepto | Implementación |
|---|---|
| Estado de partícula | `include/sph_types.h` |
| Hash espacial uniforme | `include/spatial_hash.cuh` |
| Densidad y presión | `src/kernels.cu` |
| Viscosidad y gravedad | `src/kernels.cu` |
| Fuerza y amortiguamiento SDF | `src/kernels.cu` |
| N-wave, CLI y cache | `src/main.cu` |
| Compilación | `CMakeLists.txt` |
| Producción HPC | `scripts/khipu_job_cine_sim.slurm` |

El ciclo de cada paso sigue esta secuencia:

```text
hash -> sort -> cell ranges -> density -> pressure -> forces -> integration
```

`INFORME_TECNICO.md` presenta las ecuaciones, los kernels y las decisiones de
estabilidad.

## Parámetros finales

| Parámetro | Valor |
|---|---:|
| `dt` | 0.001 s |
| `h` | 0.5 m |
| `spacing_mult` | 0.78 |
| `spacing` | 0.1053 m |
| `sdf_k` | 3000 |
| Cresta N-wave | 12.0 m |
| Lámina base | 1.0 m |
| Capacidad | 4,500,000 partículas |

## Compilación

En Linux o Khipu:

```bash
mkdir -p build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --parallel
```

En el nodo login de Khipu:

```bash
bash scripts/compile_login.sh
```

## CLI

```text
sph_tsunami [frames] [output_dir]
  [--sdf path]
  [--particles count]
  [--spacing-mult value]
  [--target-fps fps]
```

Smoke test:

```bash
./build/sph_tsunami 3 smoke_test \
  --sdf sdf/utec_sdf.bin \
  --particles 200000 \
  --target-fps 125
```

Corrida cinematográfica:

```bash
sbatch scripts/khipu_job_cine_sim.slurm
```

## Controles

El log debe mostrar:

- SDF `75x75x106`, voxel `0.250 m`
- centro del edificio con `phi < 0`
- punto lateral con `phi > 0`
- cero partículas con `phi < -0.3`
- 1,450 archivos en `cache_cine/`

La ejecución sin `--sdf` conserva el dominio de caja.
