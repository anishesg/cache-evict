#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Warp-cooperative in-place stream compaction of the KV cache.
//
// Processes entries in warp-stride order. For each group of WARP_SIZE entries,
// uses __ballot_sync to determine which entries pass the threshold, then
// computes write offsets via __popc-based prefix sum. Retained entries are
// moved forward in K, V, and importance arrays.
//
// All threads in the warp must call this function cooperatively.
// Returns the new valid_count (number of retained entries).
//
// K:          [valid_count, head_dim] float16
// V:          [valid_count, head_dim] float16
// importance: [valid_count]           float32
// threshold:  entries with importance <= threshold are evicted
// head_dim:   elements per KV vector
__device__ inline int compact_kv_cache(
    __half* __restrict__ K,
    __half* __restrict__ V,
    float* __restrict__  importance,
    int                  valid_count,
    int                  head_dim,
    float                threshold)
{
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
    int lane = threadIdx.x % 32;

    // Running write pointer shared across warp iterations.
    // Each lane tracks its local offset within its warp group; a warp-level
    // exclusive prefix sum determines the global write position.
    int write_pos = 0;  // global write cursor for the whole compaction

    for (int base = 0; base < valid_count; base += 32) {
        int pos = base + lane;
        bool keep = (pos < valid_count) && (importance[pos] > threshold);

        unsigned keep_mask = __ballot_sync(FULL_MASK, keep);

        // Exclusive prefix sum: how many lanes before this one are keeping.
        int lane_offset = __popc(keep_mask & ((1u << lane) - 1u));
        int warp_keep   = __popc(keep_mask);

        if (keep) {
            int dst = write_pos + lane_offset;
            if (dst != pos) {
                // Move K entry.
                for (int d = 0; d < head_dim; ++d) {
                    K[dst * head_dim + d] = K[pos * head_dim + d];
                }
                // Move V entry.
                for (int d = 0; d < head_dim; ++d) {
                    V[dst * head_dim + d] = V[pos * head_dim + d];
                }
                importance[dst] = importance[pos];
            }
        }
        write_pos += warp_keep;
    }

    return write_pos;
}

// Block-level compaction wrapper when multiple warps participate.
// Each warp independently compacts a slice, then results are merged.
// For simplicity and correctness, this uses a single-warp sequential pass.
// The first warp (lanes 0..31) runs the compaction; other threads synchronize.
// Returns new valid_count via shared variable; all threads see the same value.
__device__ inline int compact_kv_cache_block(
    __half* __restrict__ K,
    __half* __restrict__ V,
    float* __restrict__  importance,
    int                  valid_count,
    int                  head_dim,
    float                threshold,
    int*                 smem_new_count)  // one int of shared memory scratch
{
    int warp_id = threadIdx.x / 32;

    if (warp_id == 0) {
        int new_count = compact_kv_cache(K, V, importance, valid_count, head_dim, threshold);
        if (threadIdx.x % 32 == 0) {
            *smem_new_count = new_count;
        }
    }
    __syncthreads();
    return *smem_new_count;
}
