#pragma once
#include <cuda_fp16.h>

// Dense tiled attention reference kernel. No eviction or importance tracking.
// Operates on an explicit K/V buffer (not BoundedKVCache).
// Used as a correctness oracle.
//
// Q:      [num_heads, head_dim]  float16, current query
// K:      [seq_len, head_dim]    float16, key cache
// V:      [seq_len, head_dim]    float16, value cache
// output: [num_heads, head_dim]  float16, attention output
void launch_reference_attention(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       output,
    int           num_heads,
    int           head_dim,
    int           seq_len,
    cudaStream_t  stream = 0);
