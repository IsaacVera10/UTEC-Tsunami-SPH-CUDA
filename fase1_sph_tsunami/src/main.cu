#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(d) _mkdir(d)
#else
#include <sys/stat.h>
#define MKDIR(d) mkdir(d, 0755)
#endif

#include "sph_types.h"
#include "spatial_hash.cuh"

void upload_params(const SPHParams& p, const KernelConsts& kc);
__global__ void k_compute_hashes(const Particle* __restrict__, uint32_t* __restrict__, uint32_t* __restrict__, const SPHParams);
__global__ void k_reorder_particles(const Particle* __restrict__, Particle* __restrict__, const uint32_t* __restrict__, int);
__global__ void k_find_cell_ranges(const uint32_t* __restrict__, int* __restrict__, int* __restrict__, int, int);
__global__ void k_compute_density(Particle* __restrict__, const int* __restrict__, const int* __restrict__);
__global__ void k_compute_pressure(Particle* __restrict__);
__global__ void k_compute_forces(const Particle* __restrict__, float3* __restrict__, const int* __restrict__, const int* __restrict__);
__global__ void k_integrate(Particle* __restrict__, const float3* __restrict__);
__global__ void k_sdf_sanity();
__global__ void k_count_sdf_penetrations(const Particle* __restrict__, int* __restrict__);

#define CUDA_CHECK(call) {                                            \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

struct SDFVolume {
    float3 origin;   // esquina mín en sim-space (m)
    int3   dims;     // celdas (x, y, z) en sim-space
    float  voxel;    // m/celda
    std::vector<float> data;        // copia host (para filtrar el spawn)
    cudaArray_t         array = nullptr;
    cudaTextureObject_t tex   = 0;
};

static bool parse_sdf_meta(const std::string& meta_path, SDFVolume& sdf)
{
    FILE* f = fopen(meta_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "[sdf] no se pudo abrir %s\n", meta_path.c_str()); return false; }
    std::string txt;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) txt.append(buf, n);
    fclose(f);

    const char* o = strstr(txt.c_str(), "\"origin\"");
    const char* d = strstr(txt.c_str(), "\"dims\"");
    const char* v = strstr(txt.c_str(), "\"voxel\"");
    if (!o || !d || !v) { fprintf(stderr, "[sdf] meta sin origin/dims/voxel\n"); return false; }

    if (sscanf(strchr(o, '['), "[ %f , %f , %f", &sdf.origin.x, &sdf.origin.y, &sdf.origin.z) != 3) return false;
    if (sscanf(strchr(d, '['), "[ %d , %d , %d", &sdf.dims.x, &sdf.dims.y, &sdf.dims.z) != 3) return false;
    if (sscanf(strchr(v, ':'), ": %f", &sdf.voxel) != 1) return false;
    return true;
}

static bool load_sdf(const std::string& bin_path, SDFVolume& sdf)
{
    std::string meta_path = bin_path;
    size_t ext = meta_path.rfind(".bin");
    if (ext == std::string::npos) { fprintf(stderr, "[sdf] path sin .bin: %s\n", bin_path.c_str()); return false; }
    meta_path.replace(ext, 4, "_meta.json");

    if (!parse_sdf_meta(meta_path, sdf)) return false;

    const size_t n_vox = (size_t)sdf.dims.x * sdf.dims.y * sdf.dims.z;
    sdf.data.resize(n_vox);

    FILE* f = fopen(bin_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "[sdf] no se pudo abrir %s\n", bin_path.c_str()); return false; }
    size_t got = fread(sdf.data.data(), sizeof(float), n_vox, f);
    char probe;
    bool extra = fread(&probe, 1, 1, f) == 1;
    fclose(f);
    if (got != n_vox || extra) {
        fprintf(stderr, "[sdf] tamaño de %s no coincide con dims %dx%dx%d (leídos %zu de %zu floats)\n",
                bin_path.c_str(), sdf.dims.x, sdf.dims.y, sdf.dims.z, got, n_vox);
        return false;
    }

    cudaChannelFormatDesc ch = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(sdf.dims.z, sdf.dims.y, sdf.dims.x);
    CUDA_CHECK(cudaMalloc3DArray(&sdf.array, &ch, extent));

    cudaMemcpy3DParms cp = {};
    cp.srcPtr   = make_cudaPitchedPtr(sdf.data.data(),
                                      sdf.dims.z * sizeof(float),  // pitch en BYTES
                                      sdf.dims.z, sdf.dims.y);     // width/height en elementos
    cp.dstArray = sdf.array;
    cp.extent   = extent;
    cp.kind     = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&cp));

    cudaResourceDesc res = {};
    res.resType         = cudaResourceTypeArray;
    res.res.array.array = sdf.array;

    cudaTextureDesc td = {};
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.addressMode[2] = cudaAddressModeClamp;
    td.filterMode       = cudaFilterModeLinear;   // trilinear por hardware
    td.readMode         = cudaReadModeElementType;
    td.normalizedCoords = 0;                      // coords en texels, NO normalizadas
    CUDA_CHECK(cudaCreateTextureObject(&sdf.tex, &res, &td, nullptr));

    printf("[sdf] %s: dims %dx%dx%d, voxel %.3f m, origin (%.2f, %.2f, %.2f)\n",
           bin_path.c_str(), sdf.dims.x, sdf.dims.y, sdf.dims.z,
           sdf.voxel, sdf.origin.x, sdf.origin.y, sdf.origin.z);
    return true;
}

static float sdf_sample_host(const SDFVolume& s, float3 p)
{
    int ix = (int)roundf((p.x - s.origin.x) / s.voxel);
    int iy = (int)roundf((p.y - s.origin.y) / s.voxel);
    int iz = (int)roundf((p.z - s.origin.z) / s.voxel);
    if (ix < 0 || iy < 0 || iz < 0 ||
        ix >= s.dims.x || iy >= s.dims.y || iz >= s.dims.z)
        return 1e9f;
    return s.data[((size_t)ix * s.dims.y + iy) * s.dims.z + iz];
}

void init_tsunami_wave(std::vector<Particle>& particles, const SPHParams& p)
{
    const float spacing = p.spacing;
    const float jitter  = spacing * 0.15f;

    const float domain_x = p.domain_max.x - p.domain_min.x;
    const float domain_z = p.domain_max.z - p.domain_min.z;
    const float z_center = (p.domain_min.z + p.domain_max.z) * 0.5f;

    const float A_crest  = fminf(12.f, (p.domain_max.y - p.domain_min.y) * 0.70f);
    const float x_crest  = p.domain_min.x + domain_x * 0.12f;
    const float L_crest  = domain_x * 0.12f;

    const float A_trough = A_crest * 0.25f;
    const float x_trough = p.domain_min.x + domain_x * 0.33f;
    const float L_trough = domain_x * 0.15f;

    const float h0    = 1.0f;            // lámina en reposo (m). Iter-2: 0.5→1.0
    const float h_min = spacing * 1.2f;  // mínimo 1 capa de partículas

    srand(42);
    auto rnd = []{ return (float)rand() / (float)RAND_MAX - 0.5f; };

    particles.clear();

    for (float x = p.domain_min.x + spacing; x < p.domain_max.x - spacing; x += spacing)
    for (float z = p.domain_min.z + spacing; z < p.domain_max.z - spacing; z += spacing)
    {
        float z_norm = (z - z_center) / (domain_z * 0.5f);
        float z_mod  = 1.f - 0.15f * z_norm * z_norm;

        float dx_c = (x - x_crest)  / L_crest;
        float dx_t = (x - x_trough) / L_trough;
        float cc   = coshf(dx_c);
        float ct   = coshf(dx_t);
        float eta  = A_crest * z_mod / (cc * cc)
                   - A_trough         / (ct * ct);

        float h_local   = fmaxf(h_min, h0 + eta);
        float h_surface = p.domain_min.y + h_local;
        h_surface = fminf(h_surface, p.domain_max.y - spacing);
        if (h_surface <= p.domain_min.y + spacing * 0.5f) continue;

        float c_local = sqrtf(9.81f * h_local);
        float vx = 0.f;
        if (fabsf(eta) > 0.05f)
            vx = fmaxf(-15.f, fminf(15.f, c_local * eta / h_local));

        for (float y = p.domain_min.y + spacing; y < h_surface; y += spacing) {
            Particle part;
            part.pos      = make_float3(x + rnd()*2*jitter,
                                        y + rnd()*2*jitter,
                                        z + rnd()*2*jitter);
            part.vel      = make_float3(vx, 0.f, 0.f);
            part.density  = p.rho0;
            part.pressure = 0.f;
            particles.push_back(part);
            if ((int)particles.size() >= p.num_particles) goto done;
        }
    }
done:
    printf("[init_tsunami] N-wave: %zu part.  crest: A=%.1fm x=%.1fm L=%.1fm\n",
           particles.size(), A_crest, x_crest, L_crest);
    printf("               trough: A=%.1fm x=%.1fm L=%.1fm\n",
           A_trough, x_trough, L_trough);
}

void write_frame(FILE* f, const Particle* h_particles, int n)
{
    uint32_t nn = (uint32_t)n;
    fwrite(&nn, sizeof(uint32_t), 1, f);
    for (int i = 0; i < n; ++i) {
        fwrite(&h_particles[i].pos, sizeof(float3), 1, f);
    }
}

SPHParams make_params(int num_particles, float spacing_mult = 1.f)
{
    SPHParams p;

    p.h   = 0.5f;
    p.h2  = p.h * p.h;
    p.h3  = p.h2 * p.h;
    p.h6  = p.h3 * p.h3;
    p.h9  = p.h6 * p.h3;

    p.rho0    = 1000.f;
    p.k_stiff = 200000.f;   // knob de "dureza" del agua
    p.gamma   = 7.f;
    p.mu      = 0.05f;   // 50× viscosidad física — amortigua oscilaciones Euler
    p.gravity = make_float3(0.f, -9.81f, 0.f);
    p.dt = 0.001f;       // cumple CFL con amplio margen

#ifdef DEBUG_BUILD
    p.domain_min = make_float3(0.f, 0.f, 0.f);
    p.domain_max = make_float3(10.f, 8.f, 8.f);
#else
    p.domain_min = make_float3(0.f, 0.f, 0.f);
    p.domain_max = make_float3(60.f, 30.f, 30.f);
#endif
    p.restitution = 0.1f;

    p.spacing = p.h * 0.27f * spacing_mult;
    p.mass = p.rho0 * p.spacing * p.spacing * p.spacing;

    p.sdf_tex    = 0;
    p.sdf_origin = make_float3(0.f, 0.f, 0.f);
    p.sdf_dims   = make_int3(0, 0, 0);
    p.sdf_voxel  = 0.25f;
    p.sdf_k      = 3000.f;  // mismo orden que k_stiff de paredes (config_v2.py)

    p.num_particles = num_particles;
    p.cell_size = p.h;
    p.grid_dim  = make_int3(
        (int)ceilf((p.domain_max.x - p.domain_min.x) / p.cell_size) + 1,
        (int)ceilf((p.domain_max.y - p.domain_min.y) / p.cell_size) + 1,
        (int)ceilf((p.domain_max.z - p.domain_min.z) / p.cell_size) + 1
    );
    p.num_cells = p.grid_dim.x * p.grid_dim.y * p.grid_dim.z;
    return p;
}

KernelConsts make_kernel_consts(const SPHParams& p)
{
    KernelConsts kc;
    const float PI = 3.14159265358979f;
    kc.poly6_coeff = 315.f / (64.f * PI * p.h9);
    kc.spiky_coeff = 45.f  / (PI * p.h6);
    kc.visc_coeff  = 45.f  / (PI * p.h6);
    return kc;
}

int main(int argc, char** argv)
{
    std::string sdf_path;
    float       spacing_mult = 1.f;
    int         cli_particles = -1;
    int         target_fps = 60;
    std::vector<std::string> pos_args;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--sdf") == 0 && i + 1 < argc)              sdf_path = argv[++i];
        else if (strcmp(argv[i], "--spacing-mult") == 0 && i + 1 < argc) spacing_mult = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--particles") == 0 && i + 1 < argc)    cli_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--target-fps") == 0 && i + 1 < argc)   target_fps = atoi(argv[++i]);
        else pos_args.push_back(argv[i]);
    }

    int         num_frames  = (pos_args.size() > 0) ? atoi(pos_args[0].c_str()) : 1200;
    std::string output_dir  = (pos_args.size() > 1) ? pos_args[1] : "./cache_output";
#ifdef DEBUG_BUILD
    int         num_particles = 200000;  // dominio debug 10×8×8, spacing=0.135m → ~200k
#else
    int         num_particles = 2000000;
#endif
    if (cli_particles > 0) num_particles = cli_particles;
    if (target_fps <= 0) target_fps = 60;

    const int steps_per_frame = std::max(1, (int)std::round(1.0 / (target_fps * 0.001)));

    MKDIR(output_dir.c_str());

    SPHParams    params = make_params(num_particles, spacing_mult);
    KernelConsts kc     = make_kernel_consts(params);

    SDFVolume sdf;
    const bool use_sdf = !sdf_path.empty();
    if (use_sdf) {
        if (!load_sdf(sdf_path, sdf)) {
            fprintf(stderr, "Error cargando SDF '%s'\n", sdf_path.c_str());
            return EXIT_FAILURE;
        }
        params.sdf_tex    = sdf.tex;
        params.sdf_origin = sdf.origin;
        params.sdf_dims   = sdf.dims;
        params.sdf_voxel  = sdf.voxel;
    }

    const int total_steps   = num_frames * steps_per_frame;
    const double sim_time   = total_steps * params.dt;
    const double effective_cache_fps = 1.0 / (steps_per_frame * params.dt);

    printf("=== SPH Tsunami UTEC ===\n");
    printf("Partículas:    %d\n", num_particles);
    printf("Frames:        %d  (target %d fps, efectivo %.1f fps)\n",
           num_frames, target_fps, effective_cache_fps);
    printf("Steps/frame:   %d\n", steps_per_frame);
    printf("dt:            %.4f s\n", params.dt);
    printf("Tiempo simul:  %.2f s\n", sim_time);
    printf("h:             %.3f m\n", params.h);
    printf("spacing:       %.4f m  (spacing_mult %.3f)\n", params.spacing, spacing_mult);
    printf("Grid:          %dx%dx%d (%d celdas)\n",
           params.grid_dim.x, params.grid_dim.y, params.grid_dim.z, params.num_cells);
    printf("Output:        %s/\n", output_dir.c_str());
    printf("SDF edificio:  %s\n", use_sdf ? sdf_path.c_str() : "(sin edificio — modo v1)");
    printf("Cache size:    ~%.1f GB\n",
           (double)num_frames * num_particles * 12.0 / 1e9);
    printf("------------------------\n");

    upload_params(params, kc);

    if (use_sdf) {
        k_sdf_sanity<<<1, 1>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<Particle> h_particles;
    init_tsunami_wave(h_particles, params);

    if (use_sdf) {
        size_t before = h_particles.size();
        h_particles.erase(
            std::remove_if(h_particles.begin(), h_particles.end(),
                [&](const Particle& q){ return sdf_sample_host(sdf, q.pos) < 0.5f * sdf.voxel; }),
            h_particles.end());
        printf("[sdf] spawn: %zu partículas dentro del edificio eliminadas (%zu -> %zu)\n",
               before - h_particles.size(), before, h_particles.size());
    }

    num_particles = (int)h_particles.size();
    params.num_particles = num_particles;
    upload_params(params, kc);  // re-subir con N real

    Particle *d_particles_a, *d_particles_b;  // doble buffer para reordenamiento
    float3   *d_acc;
    uint32_t *d_hashes, *d_ids;
    int      *d_cell_start, *d_cell_end;

    size_t part_bytes = num_particles * sizeof(Particle);
    CUDA_CHECK(cudaMalloc(&d_particles_a, part_bytes));
    CUDA_CHECK(cudaMalloc(&d_particles_b, part_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc,         num_particles * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_hashes,      num_particles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ids,         num_particles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_cell_start,  params.num_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_end,    params.num_cells * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_particles_a, h_particles.data(), part_bytes, cudaMemcpyHostToDevice));

    int* d_pen_counts = nullptr;
    if (use_sdf) CUDA_CHECK(cudaMalloc(&d_pen_counts, 2 * sizeof(int)));

    std::vector<Particle> h_frame(num_particles);  // buffer host para escritura

    const int BLOCK = 256;
    int grid_n = (num_particles + BLOCK - 1) / BLOCK;

    auto t_start = std::chrono::steady_clock::now();

    for (int frame = 0; frame < num_frames; ++frame)
    {
        for (int step = 0; step < steps_per_frame; ++step)
        {
            CUDA_CHECK(cudaMemset(d_cell_start, -1, params.num_cells * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_cell_end,   -1, params.num_cells * sizeof(int)));

            k_compute_hashes<<<grid_n, BLOCK>>>(d_particles_a, d_hashes, d_ids, params);

            thrust::device_ptr<uint32_t> t_hashes(d_hashes);
            thrust::device_ptr<uint32_t> t_ids(d_ids);
            thrust::sort_by_key(t_hashes, t_hashes + num_particles, t_ids);

            k_reorder_particles<<<grid_n, BLOCK>>>(d_particles_a, d_particles_b, d_ids, num_particles);

            k_find_cell_ranges<<<grid_n, BLOCK>>>(d_hashes, d_cell_start, d_cell_end,
                                                   num_particles, params.num_cells);

            std::swap(d_particles_a, d_particles_b);

            k_compute_density<<<grid_n, BLOCK>>>(d_particles_a, d_cell_start, d_cell_end);
            k_compute_pressure<<<grid_n, BLOCK>>>(d_particles_a);

            k_compute_forces<<<grid_n, BLOCK>>>(d_particles_a, d_acc, d_cell_start, d_cell_end);

            k_integrate<<<grid_n, BLOCK>>>(d_particles_a, d_acc);
        }

        if (use_sdf) {
            CUDA_CHECK(cudaMemset(d_pen_counts, 0, 2 * sizeof(int)));
            k_count_sdf_penetrations<<<grid_n, BLOCK>>>(d_particles_a, d_pen_counts);
            int pen[2];
            CUDA_CHECK(cudaMemcpy(pen, d_pen_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost));
            printf("[sdf] frame %4d: contactos(phi<0)=%d  profundos(phi<-0.3)=%d%s\n",
                   frame, pen[0], pen[1], pen[1] > 0 ? "  <-- TUNNELING" : "");
        }

        CUDA_CHECK(cudaMemcpy(h_frame.data(), d_particles_a, part_bytes, cudaMemcpyDeviceToHost));

        char fname[512];
        snprintf(fname, sizeof(fname), "%s/frame_%06d.bin", output_dir.c_str(), frame);
        FILE* fp = fopen(fname, "wb");
        if (!fp) { fprintf(stderr, "Error abriendo %s\n", fname); continue; }
        write_frame(fp, h_frame.data(), num_particles);
        fclose(fp);

        if (frame % 50 == 0 || frame == num_frames - 1) {
            auto now = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(now - t_start).count();
            double fps  = (frame + 1) / secs;
            double eta  = (num_frames - frame - 1) / fps;
            printf("Frame %4d/%d  |  %.1f fps cache  |  ETA: %.0f s\n",
                   frame+1, num_frames, fps, eta);
        }

        CUDA_CHECK(cudaGetLastError());
    }

    auto t_end = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(t_end - t_start).count();
    printf("\nSimulación completa en %.1f s (%.1f fps promedio)\n",
           total, num_frames / total);

    cudaFree(d_particles_a); cudaFree(d_particles_b);
    cudaFree(d_acc); cudaFree(d_hashes); cudaFree(d_ids);
    cudaFree(d_cell_start); cudaFree(d_cell_end);
    if (use_sdf) {
        cudaFree(d_pen_counts);
        cudaDestroyTextureObject(sdf.tex);
        cudaFreeArray(sdf.array);
    }

    return 0;
}
