#include "sph_types.h"
#include "spatial_hash.cuh"
#include <cuda_runtime.h>
#include <cmath>

__constant__ SPHParams   d_params;
__constant__ KernelConsts d_kc;

void upload_params(const SPHParams& p, const KernelConsts& kc){
    cudaMemcpyToSymbol(d_params, &p, sizeof(SPHParams));
    cudaMemcpyToSymbol(d_kc,     &kc, sizeof(KernelConsts));
}

__global__
void k_compute_hashes(
    const Particle* __restrict__ particles,
    uint32_t* __restrict__ hashes,
    uint32_t* __restrict__ particle_ids,
    const SPHParams params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.num_particles) return;
    int3 cell = pos_to_cell(particles[i].pos, params);
    hashes[i]       = cell_hash(cell.x, cell.y, cell.z, params);
    particle_ids[i] = (uint32_t)i;
}

__global__
void k_reorder_particles(
    const Particle* __restrict__ src,
    Particle*       __restrict__ dst,
    const uint32_t* __restrict__ sorted_ids,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[sorted_ids[i]];
}

__global__
void k_find_cell_ranges(
    const uint32_t* __restrict__ sorted_hashes,
    int*            __restrict__ cell_start,
    int*            __restrict__ cell_end,
    int n, int num_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t h = sorted_hashes[i];
    if (h >= (uint32_t)num_cells) return;
    if (i == 0 || sorted_hashes[i - 1] != h)
        cell_start[h] = i;
    if (i == n - 1 || sorted_hashes[i + 1] != h)
        cell_end[h] = i + 1;
}


__device__ __forceinline__
float fetchSDF(cudaTextureObject_t tex, float gx, float gy, float gz)
{
    return tex3D<float>(tex, gz + 0.5f, gy + 0.5f, gx + 0.5f);
}

__device__
bool sdfContact(float3 pos, float* phi_out, float3* n_out)
{
    if (d_params.sdf_dims.x == 0) return false;   // sin edificio (modo v1)

    float3 o     = d_params.sdf_origin;
    int3   dims  = d_params.sdf_dims;
    float  voxel = d_params.sdf_voxel;

    float gx = (pos.x - o.x) / voxel;
    float gy = (pos.y - o.y) / voxel;
    float gz = (pos.z - o.z) / voxel;

    if (gx < 1 || gy < 1 || gz < 1 ||
        gx > dims.x - 2 || gy > dims.y - 2 || gz > dims.z - 2)
        return false;

    float phi = fetchSDF(d_params.sdf_tex, gx, gy, gz);
    if (phi >= 0.0f) return false;

    const float h = 1.0f;  // paso en celdas
    float3 grad;
    grad.x = fetchSDF(d_params.sdf_tex, gx + h, gy, gz) - fetchSDF(d_params.sdf_tex, gx - h, gy, gz);
    grad.y = fetchSDF(d_params.sdf_tex, gx, gy + h, gz) - fetchSDF(d_params.sdf_tex, gx, gy - h, gz);
    grad.z = fetchSDF(d_params.sdf_tex, gx, gy, gz + h) - fetchSDF(d_params.sdf_tex, gx, gy, gz - h);
    float len = sqrtf(grad.x*grad.x + grad.y*grad.y + grad.z*grad.z);
    if (len < 1e-6f) return false;

    float inv = 1.0f / len;
    *phi_out = phi;
    *n_out   = make_float3(grad.x * inv, grad.y * inv, grad.z * inv);
    return true;
}

__device__ __forceinline__
float3 sdfForce(float3 pos)
{
    float phi; float3 n;
    if (!sdfContact(pos, &phi, &n)) return make_float3(0, 0, 0);
    float s = d_params.sdf_k * (-phi);   // proporcional a la penetración
    return make_float3(n.x * s, n.y * s, n.z * s);
}

__global__
void k_sdf_sanity()
{
    const float3 pts[4] = {
        make_float3(45.3f,  7.3f, 14.5f),   // centro del edificio -> phi < 0
        make_float3(10.0f,  5.0f, 15.0f),   // pista de la ola     -> fuera de grilla
        make_float3(45.3f, 25.0f, 14.5f),   // encima del edificio -> phi > 0 o fuera
        make_float3(45.3f,  7.3f,  2.0f),   // al costado en Z     -> phi > 0 o fuera
    };
    printf("[sdf sanity] origin=(%.2f,%.2f,%.2f) dims=(%d,%d,%d) voxel=%.3f\n",
           d_params.sdf_origin.x, d_params.sdf_origin.y, d_params.sdf_origin.z,
           d_params.sdf_dims.x, d_params.sdf_dims.y, d_params.sdf_dims.z,
           d_params.sdf_voxel);
    for (int i = 0; i < 4; ++i) {
        float gx = (pts[i].x - d_params.sdf_origin.x) / d_params.sdf_voxel;
        float gy = (pts[i].y - d_params.sdf_origin.y) / d_params.sdf_voxel;
        float gz = (pts[i].z - d_params.sdf_origin.z) / d_params.sdf_voxel;
        if (gx < 0 || gy < 0 || gz < 0 ||
            gx > d_params.sdf_dims.x - 1 ||
            gy > d_params.sdf_dims.y - 1 ||
            gz > d_params.sdf_dims.z - 1) {
            printf("[sdf sanity] p%d (%.1f, %.1f, %.1f) -> FUERA DE GRILLA (gx=%.1f gy=%.1f gz=%.1f)\n",
                   i, pts[i].x, pts[i].y, pts[i].z, gx, gy, gz);
        } else {
            float phi = fetchSDF(d_params.sdf_tex, gx, gy, gz);
            printf("[sdf sanity] p%d (%.1f, %.1f, %.1f) -> phi = %+.3f  (%s)\n",
                   i, pts[i].x, pts[i].y, pts[i].z, phi,
                   phi < 0.f ? "DENTRO" : "fuera");
        }
    }
}

__global__
void k_count_sdf_penetrations(
    const Particle* __restrict__ particles,
    int*            __restrict__ counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_params.num_particles) return;

    float phi; float3 n;
    if (!sdfContact(particles[i].pos, &phi, &n)) return;
    atomicAdd(&counts[0], 1);
    if (phi < -0.3f) atomicAdd(&counts[1], 1);
}


__device__ __forceinline__
float W_poly6(float r2)
{
    if (r2 >= d_params.h2) return 0.f;
    float x = d_params.h2 - r2;
    return d_kc.poly6_coeff * x * x * x;
}

__device__ __forceinline__
float3 W_spiky_grad(float3 r_vec, float r)
{
    if (r <= 1e-6f || r >= d_params.h) return make_float3(0,0,0);
    float x = d_params.h - r;
    float f = d_kc.spiky_coeff * x * x / r;
    return make_float3(-f * r_vec.x, -f * r_vec.y, -f * r_vec.z);
}

__device__ __forceinline__
float W_visc_lap(float r)
{
    if (r >= d_params.h) return 0.f;
    return d_kc.visc_coeff * (d_params.h - r);
}

__global__
void k_compute_density(
    Particle*       __restrict__ particles,  // reordenadas por hash
    const int*      __restrict__ cell_start,
    const int*      __restrict__ cell_end)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_params.num_particles) return;

    float3 pi = particles[i].pos;
    int3   ci = pos_to_cell(pi, d_params);

    float rho = 0.f;
    FOREACH_NEIGHBOR(pi, ci, d_params, cell_start, cell_end, j)
        float3 pj  = particles[j].pos;
        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
        float  r2  = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
        rho += d_params.mass * W_poly6(r2);
    END_FOREACH_NEIGHBOR

    particles[i].density = rho;
}

__global__
void k_compute_pressure(Particle* __restrict__ particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_params.num_particles) return;

    float rho  = particles[i].density;
    float ratio = rho / d_params.rho0;
    float ratio2 = ratio * ratio;
    float ratio4 = ratio2 * ratio2;
    float ratio7 = ratio4 * ratio2 * ratio;
    float p = d_params.k_stiff * (ratio7 - 1.f);
    particles[i].pressure = fmaxf(0.f, p);  // sin tensión
}

__global__
void k_compute_forces(
    const Particle* __restrict__ particles,
    float3*         __restrict__ acc,          // salida: aceleración por partícula
    const int*      __restrict__ cell_start,
    const int*      __restrict__ cell_end)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_params.num_particles) return;

    float3 pi  = particles[i].pos;
    float3 vi  = particles[i].vel;
    float  ri  = particles[i].density;
    float  Pi  = particles[i].pressure;
    int3   ci  = pos_to_cell(pi, d_params);

    float3 f_pressure = make_float3(0,0,0);
    float3 f_viscosity = make_float3(0,0,0);

    FOREACH_NEIGHBOR(pi, ci, d_params, cell_start, cell_end, j)
        if (j == i) continue;

        float3 pj  = particles[j].pos;
        float3 vj  = particles[j].vel;
        float  rj  = particles[j].density;
        float  Pj  = particles[j].pressure;

        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
        float  r   = sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
        if (r >= d_params.h || r < 1e-6f) continue;

        float  pterm = (Pi/(ri*ri) + Pj/(rj*rj));
        float3 grad  = W_spiky_grad(rij, r);
        f_pressure.x -= d_params.mass * pterm * grad.x;
        f_pressure.y -= d_params.mass * pterm * grad.y;
        f_pressure.z -= d_params.mass * pterm * grad.z;

        float  lap   = W_visc_lap(r);
        float  vcoeff = d_params.mu * d_params.mass / rj * lap;
        f_viscosity.x += vcoeff * (vj.x - vi.x);
        f_viscosity.y += vcoeff * (vj.y - vi.y);
        f_viscosity.z += vcoeff * (vj.z - vi.z);
    END_FOREACH_NEIGHBOR

    float3 f_sdf = sdfForce(pi);

    float inv_rho = 1.f / ri;
    acc[i] = make_float3(
        f_pressure.x + f_viscosity.x * inv_rho + d_params.gravity.x + f_sdf.x,
        f_pressure.y + f_viscosity.y * inv_rho + d_params.gravity.y + f_sdf.y,
        f_pressure.z + f_viscosity.z * inv_rho + d_params.gravity.z + f_sdf.z
    );
}

__global__
void k_integrate(
    Particle* __restrict__ particles,
    const float3* __restrict__ acc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_params.num_particles) return;

    float3 v = particles[i].vel;
    float3 x = particles[i].pos;
    float3 a = acc[i];
    float  dt = d_params.dt;

    v.x += a.x * dt;  v.y += a.y * dt;  v.z += a.z * dt;
    x.x += v.x * dt;  x.y += v.y * dt;  x.z += v.z * dt;

    float e  = d_params.restitution;
    float3 lo = d_params.domain_min;
    float3 hi = d_params.domain_max;
    float  r  = d_params.h * 0.5f;  // margen = radio de partícula

    if (x.x < lo.x + r) { x.x = lo.x + r; if (v.x < 0) v.x *= -e; }
    if (x.x > hi.x - r) { x.x = hi.x - r; if (v.x > 0) v.x *= -e; }
    if (x.y < lo.y + r) { x.y = lo.y + r; if (v.y < 0) v.y *= -e; }
    if (x.y > hi.y - r) { x.y = hi.y - r; if (v.y > 0) v.y *= -e; }
    if (x.z < lo.z + r) { x.z = lo.z + r; if (v.z < 0) v.z *= -e; }
    if (x.z > hi.z - r) { x.z = hi.z - r; if (v.z > 0) v.z *= -e; }

    {
        float phi; float3 n;
        if (sdfContact(x, &phi, &n)) {
            float vn = v.x*n.x + v.y*n.y + v.z*n.z;
            if (vn < 0.f) {   // moviéndose hacia el interior del edificio
                float dv = (1.f + e) * vn;
                v.x -= dv * n.x;
                v.y -= dv * n.y;
                v.z -= dv * n.z;
            }
        }
    }

    const float V_MAX = 25.f;
    float v2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if (v2 > V_MAX * V_MAX) {
        float scale = V_MAX / sqrtf(v2);
        v.x *= scale;  v.y *= scale;  v.z *= scale;
    }

    particles[i].vel = v;
    particles[i].pos = x;
}
