#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdint>
#include "sph_types.h"


__device__ __forceinline__
uint32_t cell_hash(int cx, int cy, int cz, const SPHParams& p)
{
    cx = max(0, min(cx, p.grid_dim.x - 1));
    cy = max(0, min(cy, p.grid_dim.y - 1));
    cz = max(0, min(cz, p.grid_dim.z - 1));
    return (uint32_t)(cx + cy * p.grid_dim.x + cz * p.grid_dim.x * p.grid_dim.y);
}

__device__ __forceinline__
int3 pos_to_cell(float3 pos, const SPHParams& p)
{
    return make_int3(
        (int)floorf((pos.x - p.domain_min.x) / p.cell_size),
        (int)floorf((pos.y - p.domain_min.y) / p.cell_size),
        (int)floorf((pos.z - p.domain_min.z) / p.cell_size)
    );
}

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
