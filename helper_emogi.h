#ifndef HELPER_EMOGI_H_
#define HELPER_EMOGI_H_

#include <cuda.h>
#include <fstream>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <getopt.h>
#include <algorithm>
#include <sstream>
#include <utility>
#include <vector>
#include "helper_cuda.h"

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE 32

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define MEM_ALIGN_64 (~(0xfULL))
#define MEM_ALIGN_32 (~(0x1fULL))

typedef enum {
    BASELINE = 0,
    COALESCE = 1,
    COALESCE_CHUNK = 2,
} impl_type;

typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
} mem_type;

static inline bool emogi_has_suffix(const std::string &value, const char *suffix) {
    const size_t value_len = value.size();
    const size_t suffix_len = strlen(suffix);
    if (value_len < suffix_len) {
        return false;
    }
    return value.compare(value_len - suffix_len, suffix_len, suffix) == 0;
}

static inline bool emogi_is_el_file(const std::string &filename) {
    return emogi_has_suffix(filename, ".el");
}

static inline bool emogi_is_bwcsr_file(const std::string &filename) {
    return emogi_has_suffix(filename, ".bwcsr") || emogi_has_suffix(filename, ".wbcsr");
}

static inline bool emogi_is_bcsr_file(const std::string &filename) {
    return emogi_has_suffix(filename, ".bcsr") || emogi_is_bwcsr_file(filename);
}

static inline bool emogi_is_bwcsr64_file(const std::string &filename) {
    return emogi_has_suffix(filename, ".bwcsr64");
}

static inline bool emogi_is_bcsr64_file(const std::string &filename) {
    return emogi_has_suffix(filename, ".bcsr64") || emogi_is_bwcsr64_file(filename);
}

static inline bool emogi_load_el_csr(const std::string &path,
                                     std::vector<uint64_t> &vertex_list,
                                     std::vector<uint64_t> &edge_list,
                                     std::vector<double> *weight_list = NULL) {
    std::ifstream file(path.c_str());
    if (!file.is_open()) {
        fprintf(stderr, "Edge list file open failed: %s\n", path.c_str());
        return false;
    }

    std::vector<std::pair<uint64_t, uint64_t>> edges;
    std::vector<double> weights;
    uint64_t max_vertex = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        size_t begin = 0;
        while (begin < line.size() && (line[begin] == ' ' || line[begin] == '\t')) {
            begin++;
        }
        if (begin >= line.size() || line[begin] == '#' || line[begin] == '%') {
            continue;
        }

        std::istringstream iss(line.substr(begin));
        unsigned long long src = 0;
        unsigned long long dst = 0;
        double w = 1.0;
        if (!(iss >> src >> dst)) {
            continue;
        }
        if (!(iss >> w)) {
            w = 1.0;
        }

        const uint64_t src_u64 = static_cast<uint64_t>(src);
        const uint64_t dst_u64 = static_cast<uint64_t>(dst);
        edges.emplace_back(src_u64, dst_u64);
        if (weight_list != NULL) {
            weights.push_back(w);
        }
        max_vertex = std::max(max_vertex, std::max(src_u64, dst_u64));
    }
    file.close();

    const uint64_t vertex_count = edges.empty() ? 0 : (max_vertex + 1);
    const uint64_t edge_count = static_cast<uint64_t>(edges.size());
    vertex_list.assign(vertex_count + 1, 0);
    for (size_t i = 0; i < edges.size(); i++) {
        vertex_list[edges[i].first + 1]++;
    }

    for (uint64_t i = 0; i < vertex_count; i++) {
        vertex_list[i + 1] += vertex_list[i];
    }

    edge_list.resize(edge_count);
    std::vector<uint64_t> offsets(vertex_count, 0);
    for (uint64_t i = 0; i < vertex_count; i++) {
        offsets[i] = vertex_list[i];
    }

    if (weight_list != NULL) {
        weight_list->assign(edge_count, 1.0);
    }

    for (size_t i = 0; i < edges.size(); i++) {
        const uint64_t src = edges[i].first;
        const uint64_t idx = offsets[src]++;
        edge_list[idx] = edges[i].second;
        if (weight_list != NULL) {
            (*weight_list)[idx] = weights[i];
        }
    }

    return true;
}

static inline bool emogi_load_bcsr(const std::string &path,
                                   std::vector<uint64_t> &vertex_list,
                                   std::vector<uint64_t> &edge_list,
                                   std::vector<double> *weight_list = NULL) {
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Binary CSR file open failed: %s\n", path.c_str());
        return false;
    }
    const bool file_has_weight = emogi_is_bwcsr_file(path);

    uint32_t node_count = 0;
    uint32_t edge_count = 0;
    file.read(reinterpret_cast<char*>(&node_count), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&edge_count), sizeof(uint32_t));
    if (!file) {
        fprintf(stderr, "Binary CSR header read failed: %s\n", path.c_str());
        return false;
    }

    std::vector<uint32_t> row_offsets(node_count, 0);
    if (node_count > 0) {
        file.read(reinterpret_cast<char*>(row_offsets.data()), sizeof(uint32_t) * node_count);
        if (!file) {
            fprintf(stderr, "Binary CSR row offsets read failed: %s\n", path.c_str());
            return false;
        }
    }

    vertex_list.assign(static_cast<size_t>(node_count) + 1, 0);
    for (uint32_t i = 0; i < node_count; i++) {
        vertex_list[i] = row_offsets[i];
    }
    vertex_list[node_count] = edge_count;

    edge_list.assign(edge_count, 0);
    if (weight_list != NULL) {
        weight_list->assign(edge_count, 1.0);
    }

    if (file_has_weight) {
        struct emogi_edge_weighted_t {
            uint32_t end;
            uint32_t w8;
        };

        std::vector<emogi_edge_weighted_t> weighted_edges(edge_count);
        if (edge_count > 0) {
            file.read(reinterpret_cast<char*>(weighted_edges.data()), sizeof(emogi_edge_weighted_t) * edge_count);
            if (!file) {
                fprintf(stderr, "Binary CSR weighted edges read failed: %s\n", path.c_str());
                return false;
            }
        }

        for (uint32_t i = 0; i < edge_count; i++) {
            edge_list[i] = weighted_edges[i].end;
            if (weight_list != NULL) {
                (*weight_list)[i] = static_cast<double>(weighted_edges[i].w8);
            }
        }
    } else {
        std::vector<uint32_t> edges(edge_count, 0);
        if (edge_count > 0) {
            file.read(reinterpret_cast<char*>(edges.data()), sizeof(uint32_t) * edge_count);
            if (!file) {
                fprintf(stderr, "Binary CSR edges read failed: %s\n", path.c_str());
                return false;
            }
        }

        for (uint32_t i = 0; i < edge_count; i++) {
            edge_list[i] = edges[i];
        }
    }

    file.close();
    return true;
}

template <typename WeightT>
static inline bool emogi_load_bcsr_host_arrays(const std::string &path,
                                               uint64_t **vertex_list,
                                               uint64_t **edge_list,
                                               WeightT **weight_list,
                                               uint64_t *vertex_count,
                                               uint64_t *edge_count) {
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Binary CSR file open failed: %s\n", path.c_str());
        return false;
    }

    const bool file_has_weight = emogi_is_bwcsr_file(path);
    uint32_t node_count_u32 = 0;
    uint32_t edge_count_u32 = 0;
    *vertex_list = NULL;
    *edge_list = NULL;
    if (weight_list != NULL) {
        *weight_list = NULL;
    }

    file.read(reinterpret_cast<char*>(&node_count_u32), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&edge_count_u32), sizeof(uint32_t));
    if (!file) {
        fprintf(stderr, "Binary CSR header read failed: %s\n", path.c_str());
        return false;
    }

    *vertex_count = static_cast<uint64_t>(node_count_u32);
    *edge_count = static_cast<uint64_t>(edge_count_u32);

    const size_t vertex_size = (static_cast<size_t>(node_count_u32) + 1) * sizeof(uint64_t);
    const size_t edge_size = static_cast<size_t>(edge_count_u32) * sizeof(uint64_t);
    const size_t weight_size = static_cast<size_t>(edge_count_u32) * sizeof(WeightT);

    *vertex_list = (uint64_t*)malloc(vertex_size);
    *edge_list = (uint64_t*)malloc(edge_size);
    if ((node_count_u32 > 0 && *vertex_list == NULL) || (edge_count_u32 > 0 && *edge_list == NULL)) {
        fprintf(stderr, "Binary CSR host allocation failed: %s\n", path.c_str());
        free(*vertex_list);
        free(*edge_list);
        *vertex_list = NULL;
        *edge_list = NULL;
        return false;
    }

    if (weight_list != NULL) {
        *weight_list = (WeightT*)malloc(weight_size);
        if (edge_count_u32 > 0 && *weight_list == NULL) {
            fprintf(stderr, "Binary CSR weight allocation failed: %s\n", path.c_str());
            free(*vertex_list);
            free(*edge_list);
            *vertex_list = NULL;
            *edge_list = NULL;
            return false;
        }
    }

    if (node_count_u32 > 0) {
        const size_t row_chunk_size = std::min<uint64_t>(node_count_u32, 1ULL << 20);
        std::vector<uint32_t> row_offsets(row_chunk_size, 0);
        for (uint64_t base = 0; base < *vertex_count; base += row_chunk_size) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(row_chunk_size, *vertex_count - base));
            file.read(reinterpret_cast<char*>(row_offsets.data()), sizeof(uint32_t) * chunk);
            if (!file) {
                fprintf(stderr, "Binary CSR row offsets read failed: %s\n", path.c_str());
                free(*vertex_list);
                free(*edge_list);
                if (weight_list != NULL) {
                    free(*weight_list);
                    *weight_list = NULL;
                }
                *vertex_list = NULL;
                *edge_list = NULL;
                return false;
            }

            for (size_t i = 0; i < chunk; i++) {
                (*vertex_list)[base + i] = static_cast<uint64_t>(row_offsets[i]);
            }
        }
    }
    (*vertex_list)[*vertex_count] = *edge_count;

    if (file_has_weight) {
        struct emogi_edge_weighted_t {
            uint32_t end;
            uint32_t w8;
        };

        if (edge_count_u32 > 0) {
            const size_t edge_chunk_size = std::min<uint64_t>(edge_count_u32, 1ULL << 20);
            std::vector<emogi_edge_weighted_t> weighted_edges(edge_chunk_size);
            for (uint64_t base = 0; base < *edge_count; base += edge_chunk_size) {
                const size_t chunk = static_cast<size_t>(std::min<uint64_t>(edge_chunk_size, *edge_count - base));
                file.read(reinterpret_cast<char*>(weighted_edges.data()), sizeof(emogi_edge_weighted_t) * chunk);
                if (!file) {
                    fprintf(stderr, "Binary CSR weighted edges read failed: %s\n", path.c_str());
                    free(*vertex_list);
                    free(*edge_list);
                    if (weight_list != NULL) {
                        free(*weight_list);
                        *weight_list = NULL;
                    }
                    *vertex_list = NULL;
                    *edge_list = NULL;
                    return false;
                }

                for (size_t i = 0; i < chunk; i++) {
                    (*edge_list)[base + i] = static_cast<uint64_t>(weighted_edges[i].end);
                    if (weight_list != NULL) {
                        (*weight_list)[base + i] = static_cast<WeightT>(weighted_edges[i].w8);
                    }
                }
            }
        }
    } else {
        if (edge_count_u32 > 0) {
            const size_t edge_chunk_size = std::min<uint64_t>(edge_count_u32, 1ULL << 20);
            std::vector<uint32_t> edges(edge_chunk_size, 0);
            for (uint64_t base = 0; base < *edge_count; base += edge_chunk_size) {
                const size_t chunk = static_cast<size_t>(std::min<uint64_t>(edge_chunk_size, *edge_count - base));
                file.read(reinterpret_cast<char*>(edges.data()), sizeof(uint32_t) * chunk);
                if (!file) {
                    fprintf(stderr, "Binary CSR edges read failed: %s\n", path.c_str());
                    free(*vertex_list);
                    free(*edge_list);
                    if (weight_list != NULL) {
                        free(*weight_list);
                        *weight_list = NULL;
                    }
                    *vertex_list = NULL;
                    *edge_list = NULL;
                    return false;
                }

                for (size_t i = 0; i < chunk; i++) {
                    (*edge_list)[base + i] = static_cast<uint64_t>(edges[i]);
                    if (weight_list != NULL) {
                        (*weight_list)[base + i] = static_cast<WeightT>(1);
                    }
                }
            }
        }
    }

    file.close();
    return true;
}

static inline bool emogi_load_bcsr_host_arrays(const std::string &path,
                                               uint64_t **vertex_list,
                                               uint64_t **edge_list,
                                               uint64_t *vertex_count,
                                               uint64_t *edge_count) {
    return emogi_load_bcsr_host_arrays<uint32_t>(path, vertex_list, edge_list,
                                                 (uint32_t**)NULL, vertex_count, edge_count);
}

static inline bool emogi_load_bcsr64(const std::string &path,
                                     std::vector<uint64_t> &vertex_list,
                                     std::vector<uint64_t> &edge_list,
                                     std::vector<double> *weight_list = NULL) {
    fprintf(stdout, "Reading bcsr64 format: %s\n", path.c_str());
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "bcsr64 file open failed: %s\n", path.c_str());
        return false;
    }
    const bool file_has_weight = emogi_is_bwcsr64_file(path);

    uint64_t node_count = 0;
    uint64_t edge_count = 0;
    file.read(reinterpret_cast<char*>(&node_count), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&edge_count), sizeof(uint64_t));
    if (!file) {
        fprintf(stderr, "bcsr64 header read failed: %s\n", path.c_str());
        return false;
    }

    // Read uint64_t offsets directly
    vertex_list.assign(static_cast<size_t>(node_count) + 1, 0);
    if (node_count > 0) {
        file.read(reinterpret_cast<char*>(vertex_list.data()), sizeof(uint64_t) * node_count);
        if (!file) {
            fprintf(stderr, "bcsr64 row offsets read failed: %s\n", path.c_str());
            return false;
        }
    }
    vertex_list[node_count] = edge_count;

    edge_list.assign(edge_count, 0);
    if (weight_list != NULL) {
        weight_list->assign(edge_count, 1.0);
    }

    if (file_has_weight) {
        // bwcsr64: interleaved {uint64_t end, uint64_t w8} per edge
        struct emogi_edge_weighted64_t {
            uint64_t end;
            uint64_t w8;
        };

        std::vector<emogi_edge_weighted64_t> weighted_edges(edge_count);
        if (edge_count > 0) {
            file.read(reinterpret_cast<char*>(weighted_edges.data()), sizeof(emogi_edge_weighted64_t) * edge_count);
            if (!file) {
                fprintf(stderr, "bcsr64 weighted edges read failed: %s\n", path.c_str());
                return false;
            }
        }

        for (uint64_t i = 0; i < edge_count; i++) {
            edge_list[i] = weighted_edges[i].end;
            if (weight_list != NULL) {
                (*weight_list)[i] = static_cast<double>(weighted_edges[i].w8);
            }
        }
    } else {
        // bcsr64: uint64_t destinations — read directly
        if (edge_count > 0) {
            file.read(reinterpret_cast<char*>(edge_list.data()), sizeof(uint64_t) * edge_count);
            if (!file) {
                fprintf(stderr, "bcsr64 edges read failed: %s\n", path.c_str());
                return false;
            }
        }
    }

    file.close();
    return true;
}

template <typename WeightT>
static inline bool emogi_load_bcsr64_host_arrays(const std::string &path,
                                                 uint64_t **vertex_list,
                                                 uint64_t **edge_list,
                                                 WeightT **weight_list,
                                                 uint64_t *vertex_count,
                                                 uint64_t *edge_count) {
    fprintf(stdout, "Reading bcsr64 format (host arrays): %s\n", path.c_str());
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "bcsr64 file open failed: %s\n", path.c_str());
        return false;
    }

    const bool file_has_weight = emogi_is_bwcsr64_file(path);
    *vertex_list = NULL;
    *edge_list = NULL;
    if (weight_list != NULL) {
        *weight_list = NULL;
    }

    file.read(reinterpret_cast<char*>(vertex_count), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(edge_count), sizeof(uint64_t));
    if (!file) {
        fprintf(stderr, "bcsr64 header read failed: %s\n", path.c_str());
        return false;
    }

    const size_t vertex_size = (static_cast<size_t>(*vertex_count) + 1) * sizeof(uint64_t);
    const size_t edge_size = static_cast<size_t>(*edge_count) * sizeof(uint64_t);
    const size_t weight_size = static_cast<size_t>(*edge_count) * sizeof(WeightT);

    *vertex_list = (uint64_t*)malloc(vertex_size);
    *edge_list = (uint64_t*)malloc(edge_size);
    if ((*vertex_count > 0 && *vertex_list == NULL) || (*edge_count > 0 && *edge_list == NULL)) {
        fprintf(stderr, "bcsr64 host allocation failed: %s\n", path.c_str());
        free(*vertex_list);
        free(*edge_list);
        *vertex_list = NULL;
        *edge_list = NULL;
        return false;
    }

    if (weight_list != NULL) {
        *weight_list = (WeightT*)malloc(weight_size);
        if (*edge_count > 0 && *weight_list == NULL) {
            fprintf(stderr, "bcsr64 weight allocation failed: %s\n", path.c_str());
            free(*vertex_list);
            free(*edge_list);
            *vertex_list = NULL;
            *edge_list = NULL;
            return false;
        }
    }

    // Read uint64_t offsets directly
    if (*vertex_count > 0) {
        file.read(reinterpret_cast<char*>(*vertex_list), sizeof(uint64_t) * (*vertex_count));
        if (!file) {
            fprintf(stderr, "bcsr64 row offsets read failed: %s\n", path.c_str());
            free(*vertex_list);
            free(*edge_list);
            if (weight_list != NULL) { free(*weight_list); *weight_list = NULL; }
            *vertex_list = NULL;
            *edge_list = NULL;
            return false;
        }
    }
    (*vertex_list)[*vertex_count] = *edge_count;

    if (file_has_weight) {
        struct emogi_edge_weighted64_t {
            uint64_t end;
            uint64_t w8;
        };

        if (*edge_count > 0) {
            const size_t chunk_size = std::min<uint64_t>(*edge_count, 1ULL << 20);
            std::vector<emogi_edge_weighted64_t> weighted_edges(chunk_size);
            for (uint64_t base = 0; base < *edge_count; base += chunk_size) {
                const size_t chunk = static_cast<size_t>(std::min<uint64_t>(chunk_size, *edge_count - base));
                file.read(reinterpret_cast<char*>(weighted_edges.data()), sizeof(emogi_edge_weighted64_t) * chunk);
                if (!file) {
                    fprintf(stderr, "bcsr64 weighted edges read failed: %s\n", path.c_str());
                    free(*vertex_list);
                    free(*edge_list);
                    if (weight_list != NULL) { free(*weight_list); *weight_list = NULL; }
                    *vertex_list = NULL;
                    *edge_list = NULL;
                    return false;
                }

                for (size_t i = 0; i < chunk; i++) {
                    (*edge_list)[base + i] = weighted_edges[i].end;
                    if (weight_list != NULL) {
                        (*weight_list)[base + i] = static_cast<WeightT>(weighted_edges[i].w8);
                    }
                }
            }
        }
    } else {
        // Read uint64_t edges directly
        if (*edge_count > 0) {
            file.read(reinterpret_cast<char*>(*edge_list), sizeof(uint64_t) * (*edge_count));
            if (!file) {
                fprintf(stderr, "bcsr64 edges read failed: %s\n", path.c_str());
                free(*vertex_list);
                free(*edge_list);
                if (weight_list != NULL) { free(*weight_list); *weight_list = NULL; }
                *vertex_list = NULL;
                *edge_list = NULL;
                return false;
            }
            if (weight_list != NULL) {
                for (uint64_t i = 0; i < *edge_count; i++) {
                    (*weight_list)[i] = static_cast<WeightT>(1);
                }
            }
        }
    }

    file.close();
    return true;
}

static inline bool emogi_load_bcsr64_host_arrays(const std::string &path,
                                                 uint64_t **vertex_list,
                                                 uint64_t **edge_list,
                                                 uint64_t *vertex_count,
                                                 uint64_t *edge_count) {
    return emogi_load_bcsr64_host_arrays<uint32_t>(path, vertex_list, edge_list,
                                                    (uint32_t**)NULL, vertex_count, edge_count);
}

#endif
