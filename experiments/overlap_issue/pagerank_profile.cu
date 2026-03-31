/* Instrumented EMOGI PageRank: per-iteration kernel timing for zero-copy overhead analysis.
 * Outputs CSV: iter, kernel_ms, update_ms
 * Compare GPUMEM (-m 0) vs UVM_DIRECT (-m 2).
 */

#include "../../helper_emogi.h"

#define MEM_ALIGN MEM_ALIGN_64
typedef uint64_t EdgeT;
typedef float ValueT;

__global__ void initialize(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT alpha) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count) {
        value[tid] = 1.0f - alpha;
        delta[tid] = (1.0f - alpha) * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
    }
}

__global__ void kernel_coalesce(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if(warpIdx < vertex_count && label[warpIdx]) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE)
            if (i >= start)
                atomicAdd(&residual[edgeList[i]], delta[warpIdx]);

        label[warpIdx] = false;
    }
}

__global__ void update(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT tolerance, ValueT alpha, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count && residual[tid] > tolerance) {
        value[tid] += residual[tid];
        delta[tid] = residual[tid] * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
        *changed = true;
    }
}

int main(int argc, char *argv[]) {
    std::string filename;
    std::vector<uint64_t> el_vertex, el_edges, bcsr_vertex, bcsr_edges;

    bool changed_h, *changed_d, *label_d;
    int c, device = 0;
    mem_type mem = GPUMEM;
    ValueT *delta_d, *residual_d, *value_d;
    ValueT alpha = 0.85f, tolerance = 0.01f;
    uint32_t iter, max_iter = 100;
    uint64_t *vertexList_h, *vertexList_d;
    EdgeT *edgeList_h = NULL, *edgeList_d;
    uint64_t vertex_count, edge_count, vertex_size, edge_size;
    uint64_t typeT;

    float milliseconds, ms_update;
    cudaEvent_t ev_start, ev_end, ev_us, ev_ue;

    while ((c = getopt(argc, argv, "f:m:d:i:l:a:h")) != -1) {
        switch (c) {
            case 'f': filename = optarg; break;
            case 'm': mem = (mem_type)atoi(optarg); break;
            case 'd': device = atoi(optarg); break;
            case 'i': max_iter = atoi(optarg); break;
            case 'l': tolerance = atof(optarg); break;
            case 'a': alpha = atof(optarg); break;
            case 'h':
                printf("EMOGI PageRank Profiler\n\t-f input  -m mem(0/1/2)  -d device  -i max_iter  -l tol  -a alpha\n");
                return 0;
            default: break;
        }
    }
    if (filename.empty()) { fprintf(stderr, "Must specify -f <file> -m <mem>\n"); return 1; }

    checkCudaErrors(cudaSetDevice(device));
    checkCudaErrors(cudaEventCreate(&ev_start)); checkCudaErrors(cudaEventCreate(&ev_end));
    checkCudaErrors(cudaEventCreate(&ev_us)); checkCudaErrors(cudaEventCreate(&ev_ue));

    bool use_el = emogi_is_el_file(filename);
    bool use_bcsr = emogi_is_bcsr_file(filename);
    bool use_bcsr64 = emogi_is_bcsr64_file(filename);

    if (use_el) {
        if (!emogi_load_el_csr(filename, el_vertex, el_edges, NULL)) exit(1);
        vertex_count = el_vertex.size() - 1; edge_count = el_edges.size();
    } else if (use_bcsr64) {
        if (!emogi_load_bcsr64(filename, bcsr_vertex, bcsr_edges)) exit(1);
        vertex_count = bcsr_vertex.size() - 1; edge_count = bcsr_edges.size();
    } else if (use_bcsr) {
        if (!emogi_load_bcsr(filename, bcsr_vertex, bcsr_edges)) exit(1);
        vertex_count = bcsr_vertex.size() - 1; edge_count = bcsr_edges.size();
    } else {
        std::string vf = filename + ".col";
        std::ifstream f1(vf, std::ios::binary); f1.read((char*)&vertex_count,8); f1.read((char*)&typeT,8); vertex_count--;
        vertex_size = (vertex_count+1)*sizeof(uint64_t); vertexList_h = (uint64_t*)malloc(vertex_size);
        f1.read((char*)vertexList_h, vertex_size); f1.close();
        std::string ef = filename + ".dst";
        std::ifstream f2(ef, std::ios::binary); f2.read((char*)&edge_count,8); f2.read((char*)&typeT,8);
    }

    vertex_size = (vertex_count + 1) * sizeof(uint64_t);
    edge_size = edge_count * sizeof(EdgeT);

    if (use_el || use_bcsr) {
        vertexList_h = (uint64_t*)malloc(vertex_size);
        edgeList_h = (EdgeT*)malloc(edge_size);
        if (use_el) { memcpy(vertexList_h, el_vertex.data(), vertex_size); memcpy(edgeList_h, el_edges.data(), edge_size); }
        else { memcpy(vertexList_h, bcsr_vertex.data(), vertex_size); memcpy(edgeList_h, bcsr_edges.data(), edge_size); }
    }

    printf("Vertex: %lu, Edge: %lu\n", vertex_count, edge_count);
    printf("Edge array size: %.2f GB\n", edge_size / 1e9);
    printf("Memory mode: %s\n", mem == GPUMEM ? "GPUMEM" : (mem == UVM_READONLY ? "UVM_READONLY" : "UVM_DIRECT"));
    printf("alpha=%.2f, tolerance=%.4f, max_iter=%u\n", alpha, tolerance, max_iter);
    fflush(stdout);

    checkCudaErrors(cudaMalloc((void**)&label_d, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&vertexList_d, vertex_size));
    checkCudaErrors(cudaMalloc((void**)&changed_d, sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&delta_d, vertex_count * sizeof(ValueT)));
    checkCudaErrors(cudaMalloc((void**)&residual_d, vertex_count * sizeof(ValueT)));
    checkCudaErrors(cudaMalloc((void**)&value_d, vertex_count * sizeof(ValueT)));

    switch (mem) {
        case GPUMEM:
            if (!edgeList_h) {
                edgeList_h = (EdgeT*)malloc(edge_size);
                std::ifstream ef((filename+".dst").c_str(), std::ios::binary);
                uint64_t tmp; ef.read((char*)&tmp,8); ef.read((char*)&tmp,8);
                ef.read((char*)edgeList_h, edge_size); ef.close();
            }
            checkCudaErrors(cudaMalloc((void**)&edgeList_d, edge_size));
            checkCudaErrors(cudaMemcpy(edgeList_d, edgeList_h, edge_size, cudaMemcpyHostToDevice));
            break;
        case UVM_READONLY:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));
            if (edgeList_h) memcpy(edgeList_d, edgeList_h, edge_size);
            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, device));
            break;
        case UVM_DIRECT:
            checkCudaErrors(cudaMallocManaged((void**)&edgeList_d, edge_size));
            if (edgeList_h) memcpy(edgeList_d, edgeList_h, edge_size);
            checkCudaErrors(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetAccessedBy, device));
            break;
    }

    checkCudaErrors(cudaMemcpy(vertexList_d, vertexList_h, vertex_size, cudaMemcpyHostToDevice));

    uint64_t numthreads = BLOCK_SIZE;
    uint64_t numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
    uint64_t numblocks_u = ((vertex_count + numthreads) / numthreads);
    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
    dim3 blockDim_u(BLOCK_SIZE, (numblocks_u+BLOCK_SIZE)/BLOCK_SIZE);

    initialize<<<blockDim_u, numthreads>>>(label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, alpha);

    const char *mem_name = (mem == GPUMEM) ? "gpumem" : (mem == UVM_READONLY ? "uvm_ro" : "uvm_direct");
    char csv_fn[256];
    snprintf(csv_fn, sizeof(csv_fn), "pagerank_profile_%s.csv", mem_name);
    FILE *csv = fopen(csv_fn, "w");
    fprintf(csv, "iter,kernel_ms,update_ms\n");

    iter = 0;
    float total_ms = 0.0f;
    printf("Starting PageRank\n"); fflush(stdout);

    do {
        changed_h = false;
        checkCudaErrors(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaEventRecord(ev_start, 0));
        kernel_coalesce<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);
        checkCudaErrors(cudaEventRecord(ev_end, 0));

        checkCudaErrors(cudaEventRecord(ev_us, 0));
        update<<<blockDim_u, numthreads>>>(label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, tolerance, alpha, changed_d);
        checkCudaErrors(cudaEventRecord(ev_ue, 0));

        checkCudaErrors(cudaEventSynchronize(ev_ue));
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, ev_start, ev_end));
        checkCudaErrors(cudaEventElapsedTime(&ms_update, ev_us, ev_ue));

        iter++;
        total_ms += milliseconds + ms_update;
        fprintf(csv, "%u,%.4f,%.4f\n", iter, milliseconds, ms_update);
        printf("[Profile] Iter %u | Kernel %.4f ms | Update %.4f ms\n", iter, milliseconds, ms_update);
        fflush(stdout);

        checkCudaErrors(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
    } while(changed_h && iter < max_iter);

    fclose(csv);
    printf("\nTotal iterations: %u\nSum of times: %.2f ms\nCSV: %s\n", iter, total_ms, csv_fn);

    free(vertexList_h); if (edgeList_h) free(edgeList_h);
    checkCudaErrors(cudaFree(label_d)); checkCudaErrors(cudaFree(vertexList_d));
    checkCudaErrors(cudaFree(edgeList_d)); checkCudaErrors(cudaFree(changed_d));
    checkCudaErrors(cudaFree(delta_d)); checkCudaErrors(cudaFree(residual_d));
    checkCudaErrors(cudaFree(value_d));
    return 0;
}
