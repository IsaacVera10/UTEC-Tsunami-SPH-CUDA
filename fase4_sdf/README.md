# Fase 4: boundary SDF del campus

## Propósito

Esta fase convierte la geometría cerrada del campus en un campo de distancia
firmado. El solver consulta ese volumen durante cada paso.

## Homologación técnica

| Etapa | Archivo | Resultado |
|---|---|---|
| Limpieza y cierre | `01_clean_mesh.py` | `utec_clean.obj` |
| Muestreo del volumen | `02_generate_sdf.py` | `utec_sdf.bin` |
| Coordenadas | `config_v2.py` | transformación validada |
| Metadatos | `sdf/utec_sdf_meta.json` | origen, dimensiones y voxel |
| Inspección | `sdf/utec_sdf_check.obj` | isosuperficie |

## Volumen final

```text
origin = (36.0, -2.0, 1.4)
dims   = (75, 75, 106)
voxel  = 0.25 m
sign   = negative_inside
index  = (ix * dimY + iy) * dimZ + iz
```

## Ejecución

```bash
python 01_clean_mesh.py
python 02_generate_sdf.py
```

Compara `sdf/utec_sdf_check.obj` con el campus antes de una simulación completa.
El centro `(45.3, 7.3, 14.5)` debe producir un valor negativo.

`README_V2_SDF.md` conserva el registro completo de validaciones y calibración.
