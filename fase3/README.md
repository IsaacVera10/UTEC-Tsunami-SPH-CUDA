# Fase 3: escena Blender y composición

## Propósito

Esta fase combina el campus, el agua reconstruida, el spray, la iluminación y las
cámaras cinematográficas.

## Escena final

| Ajuste | Valor |
|---|---|
| Motor | Cycles |
| Backend | OptiX, CUDA como respaldo |
| Resolución | 1920 x 1080 |
| Samples | 128 |
| Denoising | activo |
| Rebotes | 8 |
| Color | AgX Medium High Contrast |

## Cámaras

| Cámara | Uso |
|---|---|
| `Cam_AereaEpic` | escala del campus y avance completo |
| `Cam_AzoteaPOV` | inundación desde la cubierta |
| `Cam_DetalleImpacto` | fachada, spray y choque |

## Construcción del blend v4

Abre una escena base que ya contenga el campus y el material `Water`. Luego aplica
el art pass:

```bash
blender -b escena_utec_base.blend \
  -P ../fase2_splashsurf/06_setup_render_scene_v4_cine.py \
  -- escena_utec_v4_cine.blend
```

Copia `escena_utec_v4_cine.blend` al directorio remoto
`fase1_sph_tsunami/` antes del render.

## Activos

Los binarios grandes quedan fuera de Git:

- `escena_utec_v4_cine.blend`
- `UTEC_RAW.glb`
- modelo texturizado del campus

Publica esos archivos en una Release o almacenamiento institucional. Registra aquí
la URL, el tamaño y el SHA-256 cuando el paquete quede disponible.

## Video

Secuencia aérea:

```bash
ffmpeg -framerate 60 -start_number 0 \
  -i render_v4_Cam_AereaEpic/frame_%06d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 17 -movflags +faststart \
  tsunami_aerea.mp4
```
