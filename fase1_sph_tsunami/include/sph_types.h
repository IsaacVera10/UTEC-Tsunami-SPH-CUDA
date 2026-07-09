#pragma once
#include <cuda_runtime.h>

struct Particle {
    float3 pos;
    float3 vel;
    float  density;
    float  pressure;
};

struct SPHParams {
    float h;          // smoothing radius (m)
    float h2;         // h^2
    float h3;         // h^3
    float h6;         // h^6
    float h9;         // h^9

    float rho0;       // densidad de reposo (kg/m³) — agua: 1000
    float k_stiff;    // rigidez de presión
    float gamma;      // exponente Tait (agua: 7)
    float mu;         // viscosidad dinámica (Pa·s) — agua: 0.001
    float mass;       // masa por partícula (kg)
    float3 gravity;   // aceleración gravitatoria

    float dt;         // timestep (s)

    float3 domain_min;
    float3 domain_max;
    float  restitution; // coeficiente de restitución en paredes

    int   num_particles;
    float spacing;    // separación inicial entre partículas (m) — solo host

    cudaTextureObject_t sdf_tex;
    float3 sdf_origin;   // esquina mín de la grilla SDF en sim-space (m)
    int3   sdf_dims;     // celdas por eje sim-space (x, y, z)
    float  sdf_voxel;    // m por celda
    float  sdf_k;        // stiffness de la fuerza de penalización

    float cell_size;  // = h
    int3  grid_dim;   // celdas en cada eje
    int   num_cells;
};

struct KernelConsts {
    float poly6_coeff;   // 315 / (64 * pi * h^9)
    float spiky_coeff;   // 45  / (pi * h^6)
    float visc_coeff;    // 45  / (pi * h^6)   (mismo)
};
