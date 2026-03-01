#ifndef HELPER_EMOGI_H_
#define HELPER_EMOGI_H_

#include <cuda.h>
#include <fstream>
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

static inline bool emogi_is_bcsr_file(const std::string &filename) {
    return emogi_has_suffix(filename, ".bcsr");
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
                                   std::vector<uint64_t> &edge_list) {
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Binary CSR file open failed: %s\n", path.c_str());
        return false;
    }

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

    std::vector<uint32_t> edges(edge_count, 0);
    if (edge_count > 0) {
        file.read(reinterpret_cast<char*>(edges.data()), sizeof(uint32_t) * edge_count);
        if (!file) {
            fprintf(stderr, "Binary CSR edges read failed: %s\n", path.c_str());
            return false;
        }
    }
    file.close();

    vertex_list.assign(static_cast<size_t>(node_count) + 1, 0);
    for (uint32_t i = 0; i < node_count; i++) {
        vertex_list[i] = row_offsets[i];
    }
    vertex_list[node_count] = edge_count;

    edge_list.resize(edge_count);
    for (uint32_t i = 0; i < edge_count; i++) {
        edge_list[i] = edges[i];
    }

    return true;
}

#endif
