#pragma once
#include <cuda_runtime.h>

// Update importance scores for positions in a single tile using EMA.
//
// attn_weights[0..tile_len) are the softmax attention weights for cache
// positions [tile_start .. tile_start + tile_len) produced during the current
// tile of the attention pass.
//
// importance[tile_start .. tile_start + tile_len) are updated in-place:
//   importance[p] = alpha * attn_weights[p - tile_start] + (1 - alpha) * importance[p]
//
// Called once per tile per head. atomicAdd is used so that updates from
// concurrent heads accumulate correctly without overwriting each other.
// After all heads complete, normalize_importance() divides by num_heads
// to recover the per-head average.
__device__ inline void update_importance_tile(
    float*       importance,
    const float* attn_weights,
    int          tile_start,
    int          tile_len,
    int          tid,
    int          block_threads,
    float        alpha)
{
    for (int i = tid; i < tile_len; i += block_threads) {
        int pos = tile_start + i;
        float w = attn_weights[i];
        // Atomic EMA update: read-modify-write with atomic float add.
        // We store the raw EMA value; normalization happens separately.
        // new_val = alpha * w + (1 - alpha) * old_val
        // = alpha * w + old_val - alpha * old_val
        // Implemented as: old + alpha * (w - old) which is not atomically safe.
        // Instead, we accumulate the weighted contribution and subtract
        // the decayed portion using a known pattern:
        //   importance[pos] *= (1 - alpha)  -- done with atomicExch pattern
        // For multi-head safety we use a simpler accumulation strategy:
        //   store sum of alpha*w across all heads, then normalize.
        // The EMA decay (1-alpha) is applied once per step in normalize_importance.
        atomicAdd(&importance[pos], alpha * w);
    }
}

// Apply EMA decay and normalize by num_heads.
// Called once per decode step, after all heads have called update_importance_tile.
// importance[p] currently holds sum of (alpha * attn_weight[p]) across all heads.
// After normalization: importance[p] = (1 - alpha) * prev_importance[p] + avg(alpha * w_p)
//
// prev_importance must be the importance array from the previous step (before any
// update_importance_tile calls). This function reads prev_importance and writes
// the final value into importance.
__device__ inline void normalize_importance(
    float*       importance,     // accumulator, updated in-place
    const float* prev_importance, // values before this step's updates
    int          valid_count,
    int          tid,
    int          block_threads,
    float        alpha,
    int          num_heads)
{
    float inv_heads = 1.0f / (float)num_heads;
    for (int p = tid; p < valid_count; p += block_threads) {
        float accumulated = importance[p] * inv_heads; // avg alpha*w across heads
        float decayed     = (1.0f - alpha) * prev_importance[p];
        importance[p]     = decayed + accumulated;
    }
}
