#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

struct ModelConfig {
    int num_heads;
    int head_dim;
    int max_cache_capacity;  // maximum KV entries to retain
    float evict_target_fraction;  // fraction of capacity to retain after eviction
    float importance_ema_alpha;   // EMA decay: importance = alpha*w + (1-alpha)*importance
    float evict_trigger_ratio;    // evict when valid_count > max_capacity * this ratio
};

// Contiguous KV cache with per-position importance scores.
// Layout: K[max_capacity * head_dim], V[max_capacity * head_dim], importance[max_capacity]
// All arrays are externally allocated; this struct holds pointers only.
struct BoundedKVCache {
    __half* K;                // [max_capacity, head_dim] row-major
    __half* V;                // [max_capacity, head_dim] row-major
    float*  importance;       // [max_capacity] float32 EMA importance per position
    int     valid_count;      // current number of valid entries (< max_capacity)
    int     max_capacity;

    // Append a new KV entry with initial importance 1.0.
    // Must only be called when valid_count < max_capacity.
    __device__ void append(const __half* k_entry, const __half* v_entry, int head_dim) {
        int pos = valid_count;
        for (int d = 0; d < head_dim; ++d) {
            K[pos * head_dim + d] = k_entry[d];
            V[pos * head_dim + d] = v_entry[d];
        }
        importance[pos] = 1.0f;
        // Atomically increment valid_count so concurrent heads see consistent state.
        atomicAdd(&valid_count, 1);
    }

    __device__ const __half* k_at(int pos, int head_dim) const {
        return K + pos * head_dim;
    }

    __device__ const __half* v_at(int pos, int head_dim) const {
        return V + pos * head_dim;
    }

    __device__ float importance_at(int pos) const {
        return importance[pos];
    }
};

// Allocate device memory for a BoundedKVCache and return a host-side struct
// with device pointers filled in.
inline BoundedKVCache alloc_cache(int max_capacity, int head_dim) {
    BoundedKVCache c;
    c.max_capacity = max_capacity;
    c.valid_count  = 0;
    cudaMalloc(&c.K,          (size_t)max_capacity * head_dim * sizeof(__half));
    cudaMalloc(&c.V,          (size_t)max_capacity * head_dim * sizeof(__half));
    cudaMalloc(&c.importance, (size_t)max_capacity * sizeof(float));
    cudaMemset(c.K,          0, (size_t)max_capacity * head_dim * sizeof(__half));
    cudaMemset(c.V,          0, (size_t)max_capacity * head_dim * sizeof(__half));
    cudaMemset(c.importance, 0, (size_t)max_capacity * sizeof(float));
    return c;
}

inline void free_cache(BoundedKVCache& c) {
    cudaFree(c.K);
    cudaFree(c.V);
    cudaFree(c.importance);
    c.K = nullptr;
    c.V = nullptr;
    c.importance = nullptr;
    c.valid_count = 0;
}
