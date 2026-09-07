#pragma once
#include <cuda_runtime.h>
#include <float.h>

static constexpr int HIST_BINS = 256;

// Compute the eviction threshold using a shared-memory importance histogram.
//
// Builds a 256-bin histogram of importance scores, then scans from the lowest
// bin to find the value that cuts to target_keep entries. Uses warp-level max
// reduction to find the range for binning.
//
// hist:         [HIST_BINS] int32 shared-memory scratch, zeroed on entry
// importance:   [valid_count] float32, read-only
// valid_count:  current cache occupancy
// target_keep:  how many entries to retain after eviction
// Returns the threshold float; entries with importance <= threshold are evicted.
//
// All threads in the block must call this cooperatively.
// Caller must zero hist[] before calling.
__device__ inline float compute_eviction_threshold(
    int*         hist,          // HIST_BINS ints in shared memory, pre-zeroed
    const float* importance,
    int          valid_count,
    int          target_keep)
{
    int tid   = threadIdx.x;
    int nthrs = blockDim.x;

    // Step 1: find max importance via warp-level reduction.
    float local_max = 0.0f;
    for (int i = tid; i < valid_count; i += nthrs) {
        local_max = fmaxf(local_max, importance[i]);
    }
    // Warp-level max.
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFu, local_max, offset));
    }
    // Block-level max via shared memory (reuse hist[0] as scratch).
    if (threadIdx.x % 32 == 0) {
        atomicMax(reinterpret_cast<int*>(hist), __float_as_int(local_max));
    }
    __syncthreads();
    float max_imp = __int_as_float(*reinterpret_cast<int*>(hist));
    // Re-zero hist[0] before building histogram.
    if (tid == 0) hist[0] = 0;
    __syncthreads();

    if (max_imp <= 0.0f) {
        // All entries have zero importance; fall back to threshold of 0 (keep all).
        return 0.0f;
    }

    // Step 2: build histogram.
    float bin_width = max_imp / (float)HIST_BINS;
    for (int i = tid; i < valid_count; i += nthrs) {
        int bin = (int)(importance[i] / bin_width);
        if (bin >= HIST_BINS) bin = HIST_BINS - 1;
        atomicAdd(&hist[bin], 1);
    }
    __syncthreads();

    // Step 3: scan histogram from lowest bin to find cutoff.
    // Only thread 0 does the scan to avoid race conditions.
    float threshold = 0.0f;
    if (tid == 0) {
        int evict_count = valid_count - target_keep;
        if (evict_count <= 0) {
            threshold = 0.0f;
        } else {
            int accumulated = 0;
            threshold = 0.0f;
            for (int b = 0; b < HIST_BINS; ++b) {
                accumulated += hist[b];
                if (accumulated >= evict_count) {
                    threshold = (b + 1) * bin_width;
                    break;
                }
            }
        }
        // Store threshold back via hist[0] for broadcast.
        hist[0] = __float_as_int(threshold);
    }
    __syncthreads();
    threshold = __int_as_float(hist[0]);

    // Re-zero hist for caller to reuse if needed.
    for (int b = tid; b < HIST_BINS; b += nthrs) hist[b] = 0;
    __syncthreads();

    return threshold;
}

// FIFO fallback: evict the oldest (lowest index) entries to reach target_keep.
// Computes the importance value at position (valid_count - target_keep) after
// sorting by position order (which is implicitly the insertion order).
// This is used when all entries have equal importance to avoid thrashing.
__device__ inline float fifo_threshold(
    const float* importance,
    int          valid_count,
    int          target_keep)
{
    // The FIFO threshold is simply: retain the last target_keep entries.
    // Entry at position (valid_count - target_keep) is the boundary.
    int evict_up_to = valid_count - target_keep;
    if (evict_up_to <= 0) return 0.0f;
    // Return the importance of the last entry that will be evicted.
    // Entries [0 .. evict_up_to-1] are evicted, [evict_up_to .. valid_count-1] kept.
    return importance[evict_up_to - 1];
}
