/* Instrumented EMOGI CC: per-iteration kernel timing for zero-copy overhead analysis.
 * Outputs CSV: iter, kernel_ms
 * Compare GPUMEM (-m 0) vs UVM_DIRECT (-m 2).
 */

#include "../../helper_emogi.h"

#define MEM_ALIGN MEM_ALIGN_64
typedef uint64_t EdgeT;

__global__ void kernel_coalesce(bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t *vertexList, EdgeT *edgeList, unsigned long long *comp, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < vertex_count && curr_visit[warpIdx] == true) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                unsigned long long comp_src = comp[warpIdx];
                const EdgeT next = edgeList[i];
                unsigned long long comp_next = comp[next];
                unsigned long long comp_target;
                EdgeT next_target;

                if (comp_next != comp_src) {
                    if (comp_src < comp_next) { next_target = next; comp_target = comp_src; }
                    else { next_target = warpIdx; comp_target = comp_next; }
                    atomicMin(&comp[next_target], comp_target);
                    next_visit[next_target] = true;
                    *changed = true;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    std::string filename;
    std::vector<uint64_t> el_vertex, el_edges, bcsr_vertex, bcsr_edges;

    bool changed_h, *changed_d, *curr_visit_d, *next_visit_d;
    int c, device = 0;
    mem_type mem = GPUMEM;
    uint32_t iter;
    unsigned long long *comp_d, *comp_h;
    uint64_t *vertexList_h, *vertexList_d;
    EdgeT *edgeList_h = NULL, *edgeList_d;
    uint64_t vertex_count, edge_count, vertex_size, edge_size;
    uint64_t typeT;

    float milliseconds;
    cudaEvent_t ev_start, ev_end;

    while ((c = getopt(argc, argv, "f:m:d:h")) != -1) {
        switch (c) {
            case 'f': filename = optarg; break;
            case 'm': mem = (mem_type)atoi(optarg); break;
            case 'd': device = atoi(optarg); break;
            case 'h':
                printf("EMOGI CC Profiler\n\t-f input  -m mem(0/1/2)  -d device\n");
                return 0;
            default: break;
        }
    }
    if (filename.empty()) { fprintf(stderr, "Must specify -f <file> -m <mem>\n"); return 1; }

    checkCudaErrors(cudaSetDevice(device));
    checkCudaErrors(cudaEventCreate(&ev_start));
    checkCudaErrors(cudaEventCreate(&ev_end));

    bool use_el = emogi_is_el_file(filename);
    bool use_bcsr = emogi_is_bcsr_file(filename);
    bool use_bcsr64 = emogi_is_bcsr64_file(filename);
    std::string vertex_file, edge_file;
    if (!use_el && !use_bcsr && !use_bcsr64) { vertex_file = filename + ".col"; edge_file = filename + ".dst"; }

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
        std::ifstream file(vertex_file.c_str(), std::ios::in | std::ios::binary);
        file.read((char*)(&vertex_count), 8); file.read((char*)(&typeT), 8); vertex_count--;
        vertex_size = (vertex_count+1) * sizeof(uint64_t);
        vertexList_h = (uint64_t*)malloc(vertex_size);
        file.read((char*)vertexList_h, vertex_size); file.close();
        std::ifstream file2(edge_file.c_str(), std::ios::in | std::ios::binary);
        file2.read((char*)(&edge_count), 8); file2.read((char*)(&typeT), 8);
        edge_size = edge_count * sizeof(EdgeT);
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
    fflush(stdout);

    // Allocate
    comp_h = (unsigned long long*)malloc(vertex_count * sizeof(unsigned long long));
    checkCudaErrors(cudaMalloc((void**)&vertexList_d, vertex_size));
    checkCudaErrors(cudaMalloc((void**)&curr_visit_d, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&next_visit_d, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**)&comp_d, vertex_count * sizeof(unsigned long long)));
    checkCudaErrors(cudaMalloc((void**)&changed_d, sizeof(bool)));

    switch (mem) {
        case GPUMEM:
            if (!edgeList_h) {
                edgeList_h = (EdgeT*)malloc(edge_size);
                std::ifstream ef(edge_file.c_str(), std::ios::in | std::ios::binary);
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

    for (uint64_t i = 0; i < vertex_count; i++) comp_h[i] = i;
    checkCudaErrors(cudaMemset(curr_visit_d, 0x01, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMemset(next_visit_d, 0x00, vertex_count * sizeof(bool)));
    checkCudaErrors(cudaMemcpy(comp_d, comp_h, vertex_count * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(vertexList_d, vertexList_h, vertex_size, cudaMemcpyHostToDevice));

    uint64_t numthreads = BLOCK_SIZE;
    uint64_t numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);

    const char *mem_name = (mem == GPUMEM) ? "gpumem" : (mem == UVM_READONLY ? "uvm_ro" : "uvm_direct");
    char csv_fn[256];
    snprintf(csv_fn, sizeof(csv_fn), "cc_profile_%s.csv", mem_name);
    FILE *csv = fopen(csv_fn, "w");
    fprintf(csv, "iter,kernel_ms\n");

    iter = 0;
    float total_ms = 0.0f;

    printf("Starting CC\n"); fflush(stdout);

    do {
        changed_h = false;
        checkCudaErrors(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaEventRecord(ev_start, 0));
        kernel_coalesce<<<blockDim, numthreads>>>(curr_visit_d, next_visit_d, vertex_count, vertexList_d, edgeList_d, comp_d, changed_d);
        checkCudaErrors(cudaEventRecord(ev_end, 0));
        checkCudaErrors(cudaEventSynchronize(ev_end));
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, ev_start, ev_end));

        checkCudaErrors(cudaMemset(curr_visit_d, 0x00, vertex_count * sizeof(bool)));
        bool *temp = curr_visit_d; curr_visit_d = next_visit_d; next_visit_d = temp;

        iter++;
        total_ms += milliseconds;
        fprintf(csv, "%u,%.4f\n", iter, milliseconds);
        printf("[Profile] Iter %u | Kernel %.4f ms\n", iter, milliseconds);
        fflush(stdout);

        checkCudaErrors(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));
    } while(changed_h);

    fclose(csv);
    printf("\nTotal iterations: %u\nSum of kernel times: %.2f ms\nCSV: %s\n", iter, total_ms, csv_fn);

    free(vertexList_h); if (edgeList_h) free(edgeList_h); free(comp_h);
    checkCudaErrors(cudaFree(vertexList_d)); checkCudaErrors(cudaFree(edgeList_d));
    checkCudaErrors(cudaFree(comp_d)); checkCudaErrors(cudaFree(curr_visit_d));
    checkCudaErrors(cudaFree(next_visit_d)); checkCudaErrors(cudaFree(changed_d));
    return 0;
}
