#pragma once
#include <cuda_fp16.h>
#include "config.cuh"

// Fused attention-eviction kernel.
//
// Performs tiled attention over BoundedKVCache with online importance scoring,
// then conditionally evicts low-importance entries and appends the new K/V.
//
// Q:          [num_heads, head_dim] float16, current query
// new_K:      [num_heads, head_dim] float16, new key to append
// new_V:      [num_heads, head_dim] float16, new value to append
// output:     [num_heads, head_dim] float16, attention output
// cache:      device pointer to BoundedKVCache (K, V, importance, valid_count)
// cfg:        model configuration
// prev_importance: copy of importance array from before this step (for EMA decay)
void launch_fused_attention_evict(
    const __half*   Q,
    const __half*   new_K,
    const __half*   new_V,
    __half*         output,
    BoundedKVCache* cache,
    const ModelConfig& cfg,
    float*          prev_importance,  // device buffer [max_capacity], caller-managed
    cudaStream_t    stream = 0);
