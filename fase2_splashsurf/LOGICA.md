# LÓGICA.md — Decisiones técnicas y correcciones

Proyecto: **Simulación fotorrealista de tsunami sobre el campus UTEC**
Curso: Computer Graphics · UTEC

Este documento explica **el porqué** de cada decisión y corrección. Para los comandos
de ejecución, ver `EJECUCION.md`.

---

## 0. Panorama del proyecto

Tres fases:

1. **Fase 1 — Reconstrucción 3D del campus.** Captura foto/video → COLMAP (SfM) →
   3D Gaussian Splatting → extracción de malla SuGaR. Entrega dos GLB:
   `UTEC_RAW.glb` (fiel pero con iluminación horneada y geometría sucia) y
   `Meshy_AI_..._texture.glb` (limpio y texturizado, aproximado).
2. **Fase 2 — Simulación SPH del fluido.** WCSPH en CUDA sobre el HPC Khipu.
3. **Fase 3 — Render.** SPH → superficie → Blender Cycles → video.

El **entregable final es un video**. Esa restricción define casi todas las decisiones
de la Fase 3.

---

## 1. Fase 2 — Correcciones de física (CUDA)

### 1.1 Bug del término de presión (el más importante)
En `kernels.cu`, la aceleración por presión se estaba dividiendo **dos veces** por la
densidad. La forma simétrica de Monaghan 1992,

```
a_presión = -Σ m_j (P_i/ρ_i² + P_j/ρ_j²) ∇W
```

**ya es una aceleración**. El código le aplicaba además un `/ρ_i` extra (heredado de la
estructura de Müller 2003, que sí lo necesita porque parte de una fuerza). Eso metía un
factor ≈ 1/1000 y **rompía la conservación de momento** (f_ij ≠ −f_ji).

**Corrección:** la presión NO se divide por `ρ_i`; solo la viscosidad (formulación de
Müller) lleva el `*inv_rho`. Validado comprobando que con el factor extra se rompe la
antisimetría de las fuerzas.

### 1.2 Rigidez de la ecuación de estado (Tait)
`k_stiff` controla la velocidad del sonido `c_s` y por tanto la compresibilidad:
- `k=3000` → c_s≈4.6 m/s → agua muy compresible (visualmente "esponjosa").
- `k=200000` → c_s≈37 m/s → Mach < 1, estable a `dt=0.001` **sin** reducir substeps. **(elegido)**
- `k≈3.2M` → <1% de variación de densidad pero exige `dt=0.0005` (doble costo).

### 1.3 Portabilidad
- `main.cu`: `mkdir(...)` (POSIX) → `std::filesystem::create_directories()` (cross-platform).
- `CMakeLists.txt`: ruta fija de CCCL → `find_package(CUDAToolkit)` + `find_path(...)`.
  Arquitectura `sm_80` (A100 de Khipu) forzada con `-DCMAKE_CUDA_ARCHITECTURES="80"`.

### 1.4 Condición inicial: N-wave (importante para la composición)
La simulación arranca con una **onda N** (cresta seguida de seno), no un dam-break:
- Dominio: **X: 0→60 m = océano→campus**. La cresta nace cerca del océano (x≈7 m) y se
  **propaga en +X hacia el campus** (x=60).
- Implicación de diseño: **el campus va en el extremo X=60**, y el surge corre todo el
  dominio hasta romper contra él.

### 1.5 Estado de los datos (histórico → actual)
El cache v1 se truncó por cuota de disco (~919 frames útiles). Superado: la corrida
**v2 final** produjo 900/900 frames @60 fps con SDF, y la **v4 cine** produjo
**1450/1450 frames @125 fps con ~3.9M partículas** (`cache_cine/`, ~78 GB) — el
dataset de producción actual. Lección operativa que quedó: vigilar `df -h` antes de
cada corrida grande y borrar caches ya respaldados.

---

## 2. Fase 3 — Decisiones de arquitectura del render

### 2.1 Blender Cycles (offline) en vez de Unity SSRF (tiempo real)
Se evaluaron ambos. Como el entregable es un **video** (no algo interactivo), Cycles gana:
el material físico de agua resuelve el look (refracción, color por profundidad, reflejos)
casi "gratis", mientras que el SSRF en Unity exigía depurar shaders a mano (el blocker
real). El trabajo de Unity/SSRF queda **documentado como respaldo**, no se descarta.

### 2.2 Composición visual (Opción A) → interacción física SDF (Opción B) — HECHA
- **Opción A (v1, histórica):** el agua se simulaba en su caja y el campus se importaba
  por separado; los edificios atravesaban el agua. Composición visual pura.
- **Opción B (v2+, IMPLEMENTADA):** el solver ahora colisiona contra un SDF del edificio
  (grilla 75×75×106, voxel 0.25 m, textura 3D CUDA con penalización `k·(−φ)·n̂` +
  amortiguamiento normal). El agua trepa, rodea y moja el edificio de verdad —
  0 penetraciones profundas en todas las corridas. Detalles matemáticos en
  `fase1_sph_tsunami/INFORME_TECNICO.md` §10; bitácora en `fase4_sdf/README_V2_SDF.md`.

El render sigue usando el modelo visual bonito (GLB Meshy); el SDF (derivado de la misma
geometría con la misma transform) es lo que siente la física. Las líneas de contacto
coinciden porque ambos vienen de la composición validada en `escena_utec_v2.blend`.

---

## 3. Reconstrucción de superficie (splashsurf)

El SPH entrega **partículas**; para renderizar agua se necesita una **malla**. Se usa
splashsurf (marching cubes sobre un campo de densidad SPH, Akinci 2012).

### 3.1 Formato PLY
splashsurf espera posiciones crudas. El primer intento generaba mallas casi vacías
(1–7 KB) por un header mal formado. El conversor legado escribía **PLY
binario** con header correcto (`binary_little_endian`, x/y/z float32) — inequívoco,
rápido y la mitad de disco que XYZ texto.

### 3.2 Unidades del smoothing-length
`smoothing-length` de splashsurf se pasa en **metros absolutos** = `RADIUS × SMOOTH`.
Un bug temprano multiplicaba mal y daba un kernel diminuto → mallas vacías.

### 3.3 SURF_SMOOTH 2.0 → 4.0 (clave para la superficie)
La superficie libre del surge sale **granulada ("cottage cheese")** porque el agua
energética separa las partículas en la superficie y el kernel de reconstrucción no las
mezcla. **El suavizado de malla (Laplaciano) no puede arreglar un campo fragmentado** — hay
que suavizar el **campo** agrandando el kernel. Subir `SURF_SMOOTH` de 2.0 a **4.0**
(soporte ~1.08 m) mezcla la granularidad de escala-partícula conservando la forma de la ola.

### 3.4 Flags de suavizado de malla
`02_splashsurf.py` añade Laplaciano ponderado (Löschner 2023) + limpieza + normales:
`--mesh-smoothing-weights=on --mesh-smoothing-iters=25 --mesh-cleanup=on --normals=on
--normals-smoothing-iters=10`. El cleanup es requisito junto al smoothing.

> **Aprendizaje:** un surge de tsunami **es** turbulento; el objetivo no es vidrio liso
> sino "agua turbulenta continua". No sobre-suavizar.

---

## 4. Escena de Blender

### 4.1 Importación y ejes
El agua (sim) es **Y-up**; Blender es **Z-up**. Se importa con
`up_axis='Y', forward_axis='NEGATIVE_Z'`, quedando: agua en `X[0,60] Y[0,30] Z[0,15]`.
El GLB del campus (glTF Y-up) lo convierte automáticamente el importador a Z-up.

### 4.2 NADA de Catmull-Clark
La malla de marching cubes **ya es densa** (~500k–1.3M vértices). El plan inicial usaba
subdivisión Catmull-Clark para suavizar el "staircase", pero subdividir ×16 una malla así
**revienta la RAM** (`std::bad_alloc` en OpenSubdiv). **Corrección:** se eliminó la
subdivisión; basta un modificador **Smooth** (Laplaciano de vértices, liviano) + shade
smooth. La subdivisión es para mallas low-poly, no para esta.

### 4.3 Material de agua
- **Glass BSDF, IOR 1.33** (refracción + Fresnel correctos), estable entre versiones.
- **Roughness 0.2** (elegido): un vidrio casi-espejo (0.02) sobre superficie con bumps
  capta mil destellos puntuales → look "cristalino/cuarzo". Subir la rugosidad difumina
  esos destellos en un brillo suave que **lee como espuma/agua agitada**.
- **Volume Absorption, densidad 0.03** (elegido): con 0.12 la ley de Beer-Lambert sobre
  decenas de metros dejaba el agua **negra opaca** (parecía un bloque sólido). Bajar a 0.03
  la vuelve translúcida y deja ver el edificio sumergido.

### 4.4 Cielo Hosek-Wilkie (no Nishita)
**Blender 5.x removió el cielo Nishita.** El enum válido es
`SINGLE_SCATTERING / MULTIPLE_SCATTERING / PREETHAM / HOSEK_WILKIE`. Se usa
**Hosek-Wilkie** (físico, estable hace años) con `sun_direction` + un Sun lamp para luz
direccional/sombras.

### 4.5 Losa de inundación (truco de VFX)
El dominio de agua (60×30) es **más chico que el edificio** (a escala 35 ≈ 66×36 m), así
que el agua simulada sola no rodea el campus. **Solución:** una **losa de agua plana y
enorme** (`--flood-z`, mismo material) que extiende la inundación hasta el horizonte. El
bloque SPH aporta el surge turbulento; la losa rodea el edificio y **oculta los bordes
rectos de la caja de simulación**. Estándar en VFX: simulas solo la zona de acción y
extiendes con geometría simple.

---

## 5. Flujo de trabajo de composición

> **El `.blend` es la fuente de verdad, no los parámetros CLI.**

Dirigir una escena 3D a ciegas por flags de texto es lento y propenso a error. Flujo final:

1. El script inicial armaba la escena **en
   `--background`** (donde los importadores de Blender 5.x sí tienen contexto válido) y la
   guarda como `.blend` con texturas empotradas.
2. Se abre ese `.blend` en la **GUI** y se posiciona campus/cámara/inundación **a mano**
   (mover `CampusRoot`, encuadrar cámara con `Ctrl+Alt+Numpad0`, vista top `Numpad 7` para
   verificar solapamiento). Se guarda con `Ctrl+S`.
3. `04_render_sequence.py` **abre ese `.blend`** y, por cada frame, reemplaza **solo la
   malla de agua**, heredando campus + cámara + inundación + luz exactamente como se
   guardaron. Por eso no hace falta traducir posiciones a parámetros.

Esto funciona porque cada frame de agua se importa en las **mismas coordenadas del
dominio**, así que la alineación con el campus se mantiene sola en toda la secuencia.

---

## 6. Aprendizajes clave (gotchas)

- **WCSPH:** la forma de Monaghan ya es aceleración; no dividir de nuevo por densidad.
- **splashsurf:** `smoothing-length` en metros (= radius × smooth); el tamaño del kernel
  (no el suavizado de malla) controla la granularidad de la superficie libre.
- **MeshLab engaña:** un sólido mate gris hace que el agua parezca roca. Juzgar SIEMPRE en
  Cycles con el material de agua, no en MeshLab.
- **Catmull-Clark** sobre malla densa de marching cubes → OOM. Usar modificador Smooth.
- **Nishita** ya no existe en Blender 5.x → Hosek-Wilkie.
- **Agua:** rugosidad ~0.2 (espuma, no cristal) + absorción ~0.03 (translúcida, no negra).
- **Losa de inundación** para extender el agua y tapar los muros de la caja.
- **GUI > CLI** para composición; el `.blend` manda en el batch.
- **N-wave:** océano en X=0, campus en X=60, ola viaja +X.

---

## 7. Limitaciones superadas y las que quedan

Superadas respecto a la lista original:
- ~~Sin colisión agua-edificio~~ → **SDF implementado y validado** (§2.2, v2+).
- ~~Cache truncado~~ → v4: 1450/1450 frames @125 fps con 3.9M partículas.
- ~~Sin espuma/whitewater~~ → **sistema de espuma propio** (§8.3): clasificación por
  vecinos + separación bulk/spray + point cloud en Cycles.
- ~~Iluminación neutra~~ → escena v4 con AgX, sol cálido, oleaje procedural, lecho marino.

Limitaciones vigentes (honestas):
- **Sin motion blur**: el intercambio de mallas por frame no tiene velocidades; a 60 fps
  el ojo lo perdona, pero un still congelado lo delata.
- **Coherencia temporal de splashsurf**: cada frame se reconstruye independiente; con las
  recetas actuales el flicker es aceptable, no nulo.
- **La caja del dominio existe**: la losa + lecho + encuadre la disimulan, pero un plano
  muy abierto la revela como "frente rectangular". Es el trade-off de simular 60×30 m.
- **Pipeline Unity/SSRF**: respaldo documentado, no mantenido.

---

## 8. Pipeline v4 CINE — decisiones y porqués (2026-07)

El render v2, correcto técnicamente, no impresionaba: cámara cerca, playback rápido,
superficie "hirviendo". El v4 atacó las tres cosas + agregó espuma. Cronología de
decisiones con su evidencia:

### 8.1 Slow-motion por densidad temporal, no por interpolación
Re-simular es barato (~20 min) → el cache se escribe a **125 fps físicos** (8 pasos
de dt=0.001 por frame, exactos) y el video se monta a 60 → 2.08× cámara lenta con
movimiento continuo real. Regla: el "peso" cinematográfico de un fluido gigante viene
del sampling temporal, no del render.

### 8.2 Más resolución SÍ, pero no para la espuma
`--spacing-mult 0.78` (4M partículas) da superficie más continua y ola más sólida.
Pero el "granulado" NO se arregla con más partículas — se arregla en reconstrucción
y render (Codex tenía razón). Las dos cosas son independientes.

### 8.3 Espuma/whitewater: clasificación + SEPARACIÓN
- **Clasificar** spray por conteo de vecinos en caja 3×3×3 (celda 2.2×spacing, numpy
  vectorizado). El umbral está en conteos → independiente de resolución.
- **Filtro de altura (`--min-y 1.6`)**: la capa superficial de un mar picado de 4M
  partículas es genuinamente dispersa — el umbral solo no distingue "superficie viva"
  de "spray volando". Lo que separa el spray dramático es que VUELA sobre el nivel
  del mar. Sin este filtro: alfombra de nieve.
- **SEPARAR antes de mallar** (el hallazgo clave): si splashsurf ve las partículas de
  spray, cada una se vuelve un blob de vidrio; si además pintas espuma encima, obtienes
  una esfera blanca DENTRO de un caparazón de vidrio ("ojo de pez"). El 07 escribe el
  PLY del bulk SIN spray, y splashsurf malla eso. El spray existe solo como espuma.
- **Render como point cloud nativo** (GN Mesh-to-Points), no esferas instanciadas:
  cientos de miles de puntos son gratis para Cycles. La decimación (--max-points,
  priorizando aislamiento+altura) es red de seguridad, no la herramienta principal.

### 8.4 El veredicto A/B/C: la espuma es una herramienta de PLANO CERCANO
Test controlado (mismo frame, aérea): sin espuma / sparse 30k / densa 236k. Ganó
**sin espuma**, por consenso doble-IA + humano. En plano general los puntos SIEMPRE
leen como textura artificial (puntillismo o estática según el radio), y sub-píxel
(r=0.012) es lo peor: el denoiser lo convierte en sal. La masa de agua limpia con su
contorno de splash es lo que lee "pesado y gigante". La espuma vive en POV/Detalle,
donde las gotas cerca de cámara venden caos (y ahí radio 0.018, con emisión 0.07
para leer blanca en sombra).

### 8.5 Integración de escena: la luz delata lo que falta
- **Lecho marino**: sin fondo bajo el agua, la losa infinita se ve pálida (la luz
  atraviesa hacia el cielo) y el dominio cerrado se ve negro (absorbe) → "isla".
  Un plano de arena a z=−0.05 iguala la profundidad óptica de ambos.
- **Losa 10 cm SOBRE el nivel del mar** (tope 2.1): tapa la costura del borde del
  dominio y el agua calmada; solo lo que sobresale (ola/splash) asoma.
- **Absorción por toma**: 0.045 es correcto de cerca pero vuelve negra una masa de
  10+ m vista desde arriba → la aérea rinde con 0.018 (override `--absorption`).
- **Encuadre mata a la caja**: el dominio debe LLENAR el cuadro; una cámara muy
  lejana convierte la sim en "sello postal flotante" y ningún material lo salva.

### 8.6 Operación en Khipu (los límites dimensionan el diseño)
- QOS `a-tesis`: `mem=98G, cpu=32, walltime=10h`
- QOS `a-pregrado`: `mem=98G, cpu=32, gpu=1, walltime=8h`
  jobs, renders serializan, `--mem>98G` = PD eterno silencioso.
- **La memoria de splashsurf escala con SMOOTH** (tamaño del kernel de densidad), no
  con CUBE: smooth 4.0 = OOM garantizado con 4M partículas; 3.2 cabe y conserva el
  look continuo (evidencia: jobs 45024/45108).
- **Concurrencia ≠ throughput**: 2 splashsurf continuous simultáneos tocan el techo de
  memoria y colapsan a 7.6/32 cores efectivos por reclaim (job 45108). Serial a 32
  hilos es más rápido. El OOM del cgroup mata SIN mensaje (.err vacío) — diagnosticar
  siempre con `sacct ... MaxRSS`.
- Login node sin internet: pysplashsurf vía wheel PyPI + shim, Blender vía tarball.

### 8.7 Regla de proceso que nos salvó: el gate de 3-5 frames
Nunca lanzar un batch de horas sin renderizar primero 3-5 frames representativos
(aproximación/impacto/clímax/calma) con el pipeline COMPLETO. Cada iteración de look
costó minutos en vez de noches, y los 5 bugs visuales (ojo de pez, gotas gigantes,
alfombra, agua negra, sello postal) se cazaron ahí y no en el batch final.
