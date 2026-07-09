# Informe Técnico — Simulación SPH de Tsunami en CUDA
## Proyecto Final — Computación Gráfica UTEC

---

## 1. Fundamentos Matemáticos

### 1.1 Smoothed Particle Hydrodynamics (SPH)

SPH discretiza un fluido continuo en partículas. Cualquier cantidad física `A` en
la posición **r** se aproxima como suma ponderada sobre partículas vecinas:

$$A(\mathbf{r}) = \sum_j \frac{m_j}{\rho_j} A_j \, W(|\mathbf{r} - \mathbf{r}_j|, h)$$

donde:
- `m_j` — masa de la partícula j
- `ρ_j` — densidad de la partícula j
- `W(r, h)` — función kernel (suavizante) con radio de influencia h
- El gradiente se transfiere al kernel: `∇A(r) = Σ (m_j/ρ_j) A_j ∇W`

### 1.2 Kernels de suavizado (Müller et al. 2003)

Se usan tres kernels distintos, cada uno optimizado para su rol:

#### Poly6 — Densidad
Suave en r=0, sin singularidad. Usado para densidad donde r=0 es frecuente.

$$W_{\text{poly6}}(r, h) = \frac{315}{64\pi h^9}(h^2 - r^2)^3 \quad \text{si } r \leq h$$

```cuda
__device__ float W_poly6(float r2) {
    if (r2 >= d_params.h2) return 0.f;
    float x = d_params.h2 - r2;
    return d_kc.poly6_coeff * x * x * x;  // poly6_coeff = 315/(64π h^9)
}
```

#### Spiky — Gradiente de presión
Tiene singularidad en r=0 (gradiente no nulo), esencial para que las partículas
se separen cuando se comprimen.

$$\nabla W_{\text{spiky}}(\mathbf{r}, h) = -\frac{45}{\pi h^6}(h - r)^2 \hat{r} \quad \text{si } r \leq h$$

```cuda
__device__ float3 W_spiky_grad(float3 r_vec, float r) {
    if (r <= 1e-6f || r >= d_params.h) return make_float3(0,0,0);
    float x = d_params.h - r;
    float f = d_kc.spiky_coeff * x * x / r;   // spiky_coeff = 45/(π h^6)
    return make_float3(-f*r_vec.x, -f*r_vec.y, -f*r_vec.z);
}
```

#### Viscosity Laplacian — Viscosidad
Laplaciano positivo garantiza que la viscosidad siempre amortigua (no amplifica).

$$\nabla^2 W_{\text{visc}}(r, h) = \frac{45}{\pi h^6}(h - r) \quad \text{si } r \leq h$$

```cuda
__device__ float W_visc_lap(float r) {
    if (r >= d_params.h) return 0.f;
    return d_kc.visc_coeff * (d_params.h - r);   // visc_coeff = 45/(π h^6)
}
```

---

## 2. Ecuaciones de Física del Fluido

### 2.1 Densidad SPH

$$\rho_i = \sum_j m_j \, W_{\text{poly6}}(|\mathbf{r}_i - \mathbf{r}_j|, h)$$

```cuda
__global__ void k_compute_density(Particle* particles, ...) {
    float rho = 0.f;
    FOREACH_NEIGHBOR(pi, ci, d_params, cell_start, cell_end, j)
        float3 rij = pi - pj;
        float  r2  = dot(rij, rij);
        rho += d_params.mass * W_poly6(r2);
    END_FOREACH_NEIGHBOR
    particles[i].density = rho;
}
```

**Condición de densidad en reposo:** para `ρ_i ≈ ρ₀` con partículas en red
uniforme de spacing `d`, la masa debe satisfacer `m = ρ₀ × d³`.

Con `spacing = h × 0.27 = 0.135 m` (h/d ≈ 3.7, ~200 vecinos en 3D):

$$m = \rho_0 \times \text{spacing}^3 = 1000 \times 0.135^3 \approx 2.46 \text{ kg}$$

Este ratio h/d = 3.7 asegura un soporte de kernel bien soportado y campo SPH
estable, con ~200 vecinos por partícula en la zona densa de la ola.

### 2.2 Ecuación de Estado de Tait (Presión)

Agua débilmente compresible (WCSPH), Monaghan 1994:

$$P_i = k_{\text{stiff}} \left[\left(\frac{\rho_i}{\rho_0}\right)^\gamma - 1\right], \quad \gamma = 7$$

con clamp `P = max(0, P)` para eliminar el artefacto de tensión superficial SPH
(partículas en superficie libre tienen `ρ < ρ₀` → P negativa → fuerza atractiva
→ catapultan hacia arriba).

```cuda
__global__ void k_compute_pressure(Particle* particles) {
    float ratio  = particles[i].density / d_params.rho0;
    float ratio2 = ratio * ratio;
    float ratio4 = ratio2 * ratio2;
    float ratio7 = ratio4 * ratio2 * ratio;
    particles[i].pressure = fmaxf(0.f, d_params.k_stiff * (ratio7 - 1.f));
}
```

**Velocidad del sonido asociada (valor de producción `k_stiff = 200 000`):**
$$c_s = \sqrt{\frac{\gamma \, k_{\text{stiff}}}{\rho_0}} = \sqrt{\frac{7 \times 200000}{1000}} \approx 37.4 \text{ m/s}$$

Para agua física real: `c_s = 1500 m/s` → `k_stiff ≈ 3.2×10⁸`, pero eso exigiría
`dt ≈ 0.00005 s` (20× más pasos). La elección `k_stiff = 200 000` es el compromiso
WCSPH clásico: mantiene Mach < 1 para las velocidades de la ola (≤15 m/s → Ma ≈ 0.4),
variación de densidad de pocos %, y es estable con `dt = 0.001` sin sub-pasos extra.
Valores bajos (p. ej. 3000, probado en desarrollo) hacen el agua visiblemente
"esponjosa/gelatinosa": la compresibilidad se nota como rebote elástico de la masa.

### 2.3 Fuerzas sobre cada partícula

#### Fuerza de Presión (Monaghan 1992, formulación simétrica)

$$\mathbf{f}_i^{\text{pres}} = -m_i \sum_j m_j \left(\frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2}\right) \nabla W_{\text{spiky}}(\mathbf{r}_{ij})$$

La formulación simétrica garantiza conservación de momentum (Newton 3a ley).

#### Fuerza de Viscosidad (Müller 2003)

$$\mathbf{f}_i^{\text{visc}} = \mu \sum_j m_j \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_j} \nabla^2 W_{\text{visc}}(r_{ij})$$

#### Aceleración total

$$\mathbf{a}_i = \frac{\mathbf{f}_i^{\text{pres}} + \mathbf{f}_i^{\text{visc}}}{\rho_i} + \mathbf{g}$$

```cuda
__global__ void k_compute_forces(const Particle* particles, float3* acc, ...) {
    FOREACH_NEIGHBOR(pi, ci, d_params, cell_start, cell_end, j)
        if (j == i) continue;
        // Presión: formulación simétrica Monaghan
        float  pterm = (Pi/(ri*ri) + Pj/(rj*rj));
        float3 grad  = W_spiky_grad(rij, r);
        f_pressure -= d_params.mass * pterm * grad;

        // Viscosidad
        float  lap    = W_visc_lap(r);
        float  vcoeff = d_params.mu * d_params.mass / rj * lap;
        f_viscosity  += vcoeff * (vj - vi);
    END_FOREACH_NEIGHBOR

    acc[i] = (f_pressure + f_viscosity) / ri + d_params.gravity;
}
```

### 2.4 Integración Temporal — Euler Simpléctica

$$\mathbf{v}^{n+1} = \mathbf{v}^n + \mathbf{a}^n \Delta t$$
$$\mathbf{x}^{n+1} = \mathbf{x}^n + \mathbf{v}^{n+1} \Delta t$$

Euler simpléctica (actualizar v antes que x) conserva energía mejor que Euler
explícito estándar. Con clamp de velocidad para estabilidad numérica:

```cuda
__global__ void k_integrate(Particle* particles, const float3* acc) {
    v += acc[i] * dt;
    x += v * dt;

    // Clamp velocidad: previene divergencia en choques supersónicos
    // 25 m/s ≈ 2.5× c_ola (c≈10 m/s para h_local=10m) — preserva la dinámica visible
    const float V_MAX = 25.f;
    float v2 = dot(v, v);
    if (v2 > V_MAX * V_MAX) v *= V_MAX / sqrtf(v2);

    // Condición de frontera: caja rígida con restitución
    if (x.y < lo.y + r) { x.y = lo.y + r; if (v.y < 0) v.y *= -e; }
    // ... (6 caras)
}
```

### 2.5 Condición CFL y Estabilidad

Para dt seguro, tres condiciones deben satisfacerse simultáneamente:

| Condición | Fórmula | Valor con params actuales |
|-----------|---------|--------------------------|
| CFL (advección/acústica) | `dt < 0.4 h / (c_s + v_max)` | `0.4×0.5/(37.4+25) ≈ 0.0032s` ✓ |
| Fuerza | `dt < 0.25 √(h/a_max)` | ✓ con los picos de aceleración observados |
| Viscosidad | `dt < 0.125 h²/(μ/ρ₀)` | `>> 1s` ✓ |

Con `dt = 0.001s` hay margen ~3× sobre la condición más restrictiva (la acústica).
El clamp `V_MAX = 25 m/s` protege ese margen contra outliers de choque.

---

## 3. Inicialización: Ola de Tsunami (N-wave sísmica, Synolakis 2002)

Un tsunami de origen sísmico no es una cresta aislada: la deformación del fondo
genera una **onda N** — una cresta seguida (o precedida) de un seno de retirada
(*drawback*, el mar que "se retira" antes del golpe). El perfil usado es la suma
de dos solitones sech²:

$$\eta(x) = A_c \, m(z)\, \text{sech}^2\!\left(\frac{x-x_c}{L_c}\right) \;-\; A_t \, \text{sech}^2\!\left(\frac{x-x_t}{L_t}\right)$$

| Parámetro | Valor (v4, calibrado) | Significado |
|-----------|----------------------|-------------|
| A_c (cresta) | **12 m** (cap `fminf(12, 0.7·Ly)`) | Amplitud — calibrada para trepar 11-13.5 m en la fachada |
| x_c, L_c | 0.12·Lx = 7.2 m | Posición y ancho de la cresta (borde oceánico) |
| A_t (seno) | 0.25·A_c = 3 m | Drawback delante de la cresta |
| x_t, L_t | 0.33·Lx, 0.15·Lx | Posición/ancho del seno |
| m(z) | `1 − 0.15·ẑ²` | Modulación transversal: cresta más alta al centro |
| h₀ | **1.0 m** | Lámina en reposo — masa de impacto y menos disipación en la pista |

La superficie libre es `h_surface(x,z) = h₀ + η(x,z)`; las partículas llenan la
columna con jitter aleatorio (`srand(42)`, amplitud 0.15·spacing) para romper la
red cristalina. La lámina h₀ garantiza soporte de kernel en todo el dominio.

**Velocidad inicial (teoría de aguas someras):**

$$v_x = c_{\text{local}} \cdot \frac{\eta}{h_0 + \eta}, \qquad c_{\text{local}} = \sqrt{g(h_0 + \eta)}, \qquad |v_x| \le 15 \text{ m/s}$$

El signo de η da el sentido: +x bajo la cresta, −x bajo el seno (el drawback
succiona hacia el mar). Solo se aplica donde `|η| > 0.05 m`.

**Calibración (validada midiendo el cache):** con A_c=9/h₀=0.5 la ola trepaba solo
~4.8 m (p95) contra la fachada; con **A_c=12/h₀=1.0** alcanza 11–13.5 m de lengua
coherente en el pico del impacto (medido por percentiles de altura por franja de
1 m a lo largo de la fachada — el percentil agregado engaña porque la piscina al
pie lo arrastra).

**Partículas generadas:** el conteo escala con `1/spacing³`. Con el perfil actual:
- v2: spacing 0.135 m → 1 908 492 generadas → 1 866 488 tras el filtro de spawn (§10.5)
- v4: spacing 0.1053 m (`--spacing-mult 0.78`) → 4 022 960 → 3 922 371

---

## 4. Estructura de Datos y Spatial Hash

### 4.1 Problema: búsqueda de vecinos O(N²) → O(N)

Para cada partícula i, buscar todas j con `|r_ij| < h` en N≈2M partículas
sería O(N²) = 4×10¹² operaciones/frame. Inaceptable.

### 4.2 Spatial Hash Grid

División del dominio en celdas de tamaño `h`. Solo las 27 celdas vecinas (3×3×3)
pueden contener partículas dentro del radio h.

```
celda(x,y,z) = floor((pos - domain_min) / h)
hash(cx,cy,cz) = cx + cy×Gx + cz×Gx×Gy     (índice plano, con clamp)
```

Para el dominio UTEC (60×30×30m) con h=0.5:
- Grid: 121×61×61 = **450 241 celdas**
- Cada celda: ≈ 9 partículas en promedio (con ~3.9M partículas)

### 4.3 Pipeline por frame (GPU)

```
1. k_compute_hashes    ← asigna hash a cada partícula        O(N)
2. thrust::sort_by_key ← ordena por hash (radix sort GPU)    O(N log N)
3. k_reorder_particles ← reordena para coalescencia memoria  O(N)
4. k_find_cell_ranges  ← construye cell_start[], cell_end[]  O(N)
5. k_compute_density   ← Σ m·W_poly6                        O(N·27)
6. k_compute_pressure  ← Tait EOS                            O(N)
7. k_compute_forces    ← Σ presión + viscosidad              O(N·27)
8. k_integrate         ← Euler simpléctica + BC              O(N)
9. cudaMemcpy D→H      ← copiar posiciones                   O(N)
10. write_frame        ← escribir .bin                        O(N)
```

El reordenamiento (paso 3) es crucial: con partículas ordenadas por hash,
los accesos a vecinos son coalescentes en memoria GPU (mismo warp accede
a direcciones contiguas) → ~3× speedup sobre acceso aleatorio.

### 4.4 Macro FOREACH_NEIGHBOR

```cuda
#define FOREACH_NEIGHBOR(pos_i, cell_i, p, cs, ce, j)              \
    for (int _dz = -1; _dz <= 1; ++_dz)                            \
    for (int _dy = -1; _dy <= 1; ++_dy)                            \
    for (int _dx = -1; _dx <= 1; ++_dx) {                          \
        int _nx = (cell_i).x + _dx;                                \
        int _ny = (cell_i).y + _dy;                                \
        int _nz = (cell_i).z + _dz;                                \
        if (_nx < 0 || _nx >= (p).grid_dim.x ||                    \
            _ny < 0 || _ny >= (p).grid_dim.y ||                    \
            _nz < 0 || _nz >= (p).grid_dim.z) continue;           \
        uint32_t _hkey = cell_hash(_nx, _ny, _nz, p);              \
        int _start = (cs)[_hkey];                                   \
        int _end   = (ce)[_hkey];                                   \
        if (_start < 0) continue;                                   \
        for (int j = _start; j < _end; ++j) {
#define END_FOREACH_NEIGHBOR }}
```

Las comprobaciones de frontera (`_nx < 0 || ...`) reemplazan el clamp de hash,
evitando que celdas fuera del dominio mapeen a índices válidos.

---

## 5. Arquitectura CUDA

### 5.1 Organización de kernels

```
kernels.cu  ← toda la física GPU
│
├── __constant__ SPHParams   d_params   ← broadcast sin penalización
├── __constant__ KernelConsts d_kc
│
├── k_compute_hashes(particles, hashes, ids, params)
├── k_reorder_particles(src, dst, sorted_ids, n)
├── k_find_cell_ranges(sorted_hashes, cell_start, cell_end, n)
├── k_compute_density(particles, cell_start, cell_end)
├── k_compute_pressure(particles)
├── k_compute_forces(particles, acc, cell_start, cell_end)
└── k_integrate(particles, acc)
```

### 5.2 Constant Memory y Separable Compilation

`d_params` y `d_kc` viven en `__constant__` memory: latencia de lectura ~4 ciclos
(vs ~200 ciclos para global memory). Al ser accedidos por todos los threads del
mismo warp en broadcast, el costo es efectivamente gratuito.

**Bug crítico resuelto:** sin `-rdc=true`, cada `.cu` tiene su propia copia privada
de los símbolos `__constant__`. `upload_params()` en `kernels.cu` escribía en su
copia, pero `main.cu` (sin RDC) podría referenciar otra copia no inicializada.

**Fix en CMakeLists.txt:**
```cmake
set_property(TARGET sph_tsunami PROPERTY CUDA_SEPARABLE_COMPILATION ON)
```

### 5.3 Doble Buffer para Reordenamiento

```
Frame N:  d_particles_a  →  hash/sort  →  reorder  →  d_particles_b
                                                              ↓
                                                         swap pointers
Frame N+1: d_particles_a (apunta al buffer ordenado)
```

Evita copias innecesarias: solo se intercambian punteros (`std::swap`).

### 5.4 Sub-stepping y densidad temporal del cache (`--target-fps`)

```cpp
// target_fps es flag CLI; steps_per_frame = round(1 / (target_fps * dt))
// target_fps=60  -> 17 pasos/frame (58.8 Hz efectivo)  — playback tiempo real
// target_fps=125 ->  8 pasos/frame (125 Hz EXACTO)     — slow-motion cine
for (int frame = 0; frame < num_frames; ++frame) {
    for (int step = 0; step < steps_per_frame; ++step)
        simulate_one_step();   // dt = 0.001 s
    write_frame();
}
```

**Slow-motion real por densidad temporal (v4):** el cache se escribe a 125 fps
físicos y el video se ensambla a 60 fps → cámara lenta **2.08×** sin interpolar
frames. 1450 frames × 8 ms = 11.6 s de física = 24.2 s de video. La alternativa
(reproducir un cache de 60 fps a 30) duplica cada frame y se ve entrecortada;
la densidad temporal da movimiento continuo real.

---

## 6. Parámetros de Producción

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `h` (smoothing radius) | 0.5 m | Radio de influencia del kernel |
| `spacing` | h×0.27×mult (v2: 0.135, v4: 0.1053) | `--spacing-mult` escala la resolución; masa se recalcula |
| `mass` | ρ₀×spacing³ (v2: 2.46 kg, v4: 1.17 kg) | Densidad en reposo ≈ ρ₀ para red uniforme |
| `rho0` | 1000 kg/m³ | Densidad del agua |
| `k_stiff` | **200 000 Pa** | c_s≈37 m/s, Mach<1; valores bajos = agua gelatinosa |
| `gamma` | 7 | Exponente Tait estándar para agua |
| `mu` | 0.05 Pa·s | 50× viscosidad física; amortiguación numérica de Euler |
| `dt` | 0.001 s | Margen 3× sobre CFL acústica (c_s+V_MAX) |
| `gravity` | (0, -9.81, 0) m/s² | Gravedad estándar |
| `restitution` | 0.1 | Paredes/SDF casi inelásticos (menos rebote energético) |
| `V_MAX` | 25 m/s | Clamp anti-divergencia; ≈2.3× la c de aguas someras del frente |
| `--target-fps` | 60 (real) / 125 (slow-mo) | Densidad temporal del cache (§5.4) |
| `sdf_k` | 3000 | Rigidez de la penalización contra el edificio (§10) |

### Dominio de producción (v2/v4)

| Parámetro | Valor |
|-----------|-------|
| Dominio | 60 × **30** × 30 m (techo 2× la altura del edificio, para splashes) |
| Condición inicial | N-wave: A_c=12m, A_t=3m, h₀=1.0m (§3, calibrada) |
| Grid hash | 121 × 61 × 61 = 450 241 celdas |
| Partículas | v2: 1.87M · v4: 3.92M (tras filtro de spawn) |
| Frames de cache | v2: 900 @60fps (15.3 s) · v4: 1450 @125fps (11.6 s) |
| Edificio | SDF 75×75×106, voxel 0.25 m (§10) |

### Dominio de debug (compilado con `-DDEBUG_BUILD`)

| Parámetro | Valor |
|-----------|-------|
| Dominio | 10 × 8 × 8 m |
| Límite de partículas | 50 000 |
| Partículas reales | ~30 000 |

---

## 7. Formato del Cache Binario

```
frame_XXXXXX.bin:
  [uint32_t N        ]  ← número de partículas (4 bytes)
  [float x₀, y₀, z₀ ]  ← posición partícula 0 (12 bytes)
  [float x₁, y₁, z₁ ]  ← posición partícula 1 (12 bytes)
  ...
  [float xₙ, yₙ, zₙ ]  ← posición partícula N-1
```

Tamaño por frame: `4 + N × 12 ≈ 4 + 2 000 000 × 12 ≈ 24 MB`
Total 1200 frames: `≈ 28.8 GB` (sin comprimir)

**Lectura en C# (Unity):**
```csharp
using var br = new BinaryReader(File.OpenRead(path));
uint n = br.ReadUInt32();
var positions = new Vector3[n];
for (int i = 0; i < n; i++)
    positions[i] = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
```

---

## 8. Resultados y Rendimiento

### 8.1 Rendimiento en GPU

| Corrida | GPU | Partículas | Pasos/frame | fps cache | Tiempo total |
|---------|-----|-----------|-------------|-----------|--------------|
| v2 final (900 fr) | A100 40GB | 1 866 488 | 17 | 3.7 | 80.7 s |
| v4 cine (1450 fr) | A100 40GB | 3 922 371 | 8 | 1.1–1.2 | ~21 min |
| debug local | RTX 4060 Laptop | ~150 000 | 17 | ~45 | ~7 s |

Costo por paso ≈ O(N·vecinos): v4 duplica N y (con h fijo y spacing menor) más
que duplica los vecinos por partícula, de ahí la caída de throughput. El sort
Thrust y los kernels de fuerzas dominan el tiempo de frame.

### 8.2 Evolución de la Ola (v4, medida por el contador de contactos con el SDF)

| Tiempo | Frame @125fps | Estado |
|--------|---------------|--------|
| 0.0 s | 0 | N-wave inicial con velocidades shallow-water |
| ~1.5 s | ~190 | El frente golpea la fachada (despegue de contactos) |
| ~3.3 s | ~420 | Pico de trepada en la fachada (11–13.5 m) |
| ~6.0 s | ~760 | Clímax de acumulación (~71k partículas en contacto) |
| ~10.5 s | ~1330 | Azotea y dominio en calma (spray ≈ 0) |

La velocidad del frente (~10-15 m/s) es consistente con aguas someras
`c ≈ √(g·H)`, y el timing escala igual entre v2 (60 fps) y v4 (125 fps) —
la resolución no alteró la dinámica macroscópica.

---

## 9. Workflow Completo

### Debug local

```bash
cd build
make sph_debug -j4
./sph_debug 200 test_output      # dominio 10×8×8m, ~30k partículas
python3 ../visualize_cache.py    # genera PNGs
```

### Producción en Khipu (A100/RTX A6000)

```bash
# 1. Compilar en nodo LOGIN (tiene /usr/include)
bash scripts/compile_login.sh

# 2. Enviar job al nodo GPU (1200 frames, ~2M partículas)
sbatch scripts/khipu_job.slurm

# 3. Descargar cache
scp usuario@khipu.utec.edu.pe:~/computer_graphics/fase1_sph_tsunami/cache_tsunami.tar.gz .
tar -xzf cache_tsunami.tar.gz -C output/
```

### Visualización local

```bash
python3 visualize_cache.py              # auto-detecta ./output/
python3 visualize_cache.py ./output     # ruta explícita
```

---

## 10. Colisión con el Edificio — Boundary por SDF

La condición de frontera contra el edificio UTEC usa un **Signed Distance Field**
precomputado (fase4_sdf): una grilla 3D donde cada voxel guarda la distancia con
signo a la superficie del edificio (`φ < 0` = dentro, `φ > 0` = fuera).

### 10.1 El SDF como textura 3D CUDA

Grilla: **75×75×106 voxels de 0.25 m**, origen sim (36, −2, 1.4) — cubre el bbox
del edificio + 2 m de margen, ~2.4 MB en float32. No cabe en `__constant__`
(64 KB totales), así que vive en un `cudaArray` 3D + `cudaTextureObject_t` con
**filtrado trilineal por hardware** (φ continuo entre voxels) y clamp en bordes.

**Convención de ejes (la trampa clásica):** el `.bin` está en orden x-lento/z-rápido
(`idx = (ix·dimY + iy)·dimZ + iz`) y `cudaMemcpy3D` exige que `extent.width` sea el
eje más rápido en memoria. El volumen se sube SIN transponer (`width=dimZ`) y el
fetch compensa en UN solo lugar:

```cuda
__device__ float fetchSDF(cudaTextureObject_t t, float gx, float gy, float gz) {
    return tex3D<float>(t, gz + 0.5f, gy + 0.5f, gx + 0.5f);  // (z,y,x) + centro de texel
}
```

Un test de sanidad (`k_sdf_sanity`, corre en cada arranque) evalúa φ en 4 puntos
conocidos; el centro del edificio debe dar φ = −0.057.

### 10.2 Fuerza de penalización

En `k_compute_forces`, cada partícula que penetra (φ < 0) recibe una aceleración
proporcional a la penetración, en la dirección de la normal exterior:

$$\mathbf{a}_{\text{sdf}} = k_{\text{sdf}} \,(-\varphi)\, \hat{n}, \qquad \hat{n} = \frac{\nabla\varphi}{|\nabla\varphi|}, \qquad k_{\text{sdf}} = 3000$$

El gradiente se calcula por **diferencias centrales con 6 fetches separados** —
NUNCA con el gradiente implícito del filtrado por hardware: la interpolación
trilineal usa pesos de ~9 bits y produce normales escalonadas/ruidosas.

```cuda
grad.x = fetchSDF(tex, gx+1, gy, gz) - fetchSDF(tex, gx-1, gy, gz);  // etc. y,z
```

### 10.3 Amortiguamiento de velocidad normal

La penalización sola produce rebote elástico ("gelatina"). En `k_integrate`, si la
partícula penetra y avanza hacia el interior, se refleja la componente normal con
la misma restitución de las paredes:

$$v_n = \mathbf{v}\cdot\hat{n} < 0 \;\Rightarrow\; \mathbf{v} \mathrel{-}= (1+e)\, v_n\, \hat{n}, \qquad e = 0.1$$

### 10.4 Verificación continua (contador de penetraciones)

Cada frame, un kernel de reducción cuenta contactos (φ<0) y **penetraciones
profundas (φ<−0.3 ≈ tunneling)**. Resultado en todas las corridas de producción
(900×1.87M y 1450×3.92M): **0 penetraciones profundas, 0 NaN**. La curva de
contactos, además, sirve de instrumento: da el timing del impacto (§8.2).

### 10.5 Filtro de spawn

La lámina base h₀ cubre todo el piso — incluido el footprint del edificio. Sin
filtro, esas partículas nacen con φ ≪ 0 y la penalización las eyecta (explosión
numérica). Antes de subir a GPU se eliminan en CPU las partículas con
φ < 0.5·voxel (muestreo nearest del SDF): −42 004 partículas en v4.

### 10.6 Retrocompatibilidad

Sin `--sdf`, `sdf_dims.x == 0` y todos los caminos SDF devuelven fuerza cero —
el solver corre idéntico a la versión de caja simple.

---

## 11. Playback y render (resumen)

El cache `.bin` alimenta el pipeline de la fase 2/3 (ver `fase2_splashsurf/LOGICA.md`
y `EJECUCION.md`): separación bulk/spray (espuma), reconstrucción de superficie con
splashsurf (marching cubes sobre el campo de densidad, dos recetas), y render en
Blender Cycles con intercambio de malla por frame sobre una escena-plantilla.
