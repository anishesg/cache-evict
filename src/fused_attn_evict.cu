#include "fused_attn_evict.cuh"
#include "config.cuh"
#include "importance.cuh"
#include "compaction.cuh"
#include "threshold.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstring>

static constexpr int FTILE = 64;  // KV tile size for fused kernel

// Shared memory layout (allocated by host):
//   float K_tile[FTILE * head_dim]
//   float V_tile[FTILE * head_dim]
//   float attn_weights[FTILE]          -- per-tile softmax weights for importance update
//   float prev_imp_smem[max_capacity]  -- snapshot of importance before updates
//   int   hist[HIST_BINS]              -- histogram for threshold computation
//   int   smem_new_count               -- scratch for compaction result
//
// One block per attention step (all heads cooperative within the block).
// blockDim.x = head_dim (up to 128).
__global__ void fused_attention_evict_kernel(
    const __half* __restrict__ Q,        // [num_heads, head_dim]
    const __half* __restrict__ new_K,    // [num_heads, head_dim]
    const __half* __restrict__ new_V,    // [num_heads, head_dim]
    __half*       __restrict__ output,   // [num_heads, head_dim]
    __half*       __restrict__ cache_K,  // [max_capacity, head_dim]
    __half*       __restrict__ cache_V,  // [max_capacity, head_dim]
    float*        __restrict__ importance,
    int*          __restrict__ valid_count_ptr,
    const float*  __restrict__ prev_importance,
    int   num_heads,
    int   head_dim,
    int   max_capacity,
    float evict_trigger_ratio,
    float evict_target_fraction,
    float alpha)
{
    int tid   = threadIdx.x;
    int nthrs = blockDim.x;

    extern __shared__ float smem_float[];
    float* K_tile      = smem_float;
    float* V_tile      = K_tile  + FTILE * head_dim;
    float* w_tile      = V_tile  + FTILE * head_dim;  // [FTILE] attn weights
    float* acc_buf     = w_tile  + FTILE;              // [num_heads * head_dim] output accumulators
    float* max_buf     = acc_buf + num_heads * head_dim; // [num_heads] running max
    float* sum_buf     = max_buf + num_heads;           // [num_heads] running sum
    int*   hist        = reinterpret_cast<int*>(sum_buf + num_heads); // [HIST_BINS]
    int*   smem_nc     = hist + HIST_BINS;              // [1] new_count scratch

    int valid_count = *valid_count_ptr;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Initialize per-head accumulators.
    for (int i = tid; i < num_heads * head_dim; i += nthrs) acc_buf[i] = 0.0f;
    for (int h = tid; h < num_heads; h += nthrs) {
        max_buf[h] = -FLT_MAX;
        sum_buf[h] = 0.0f;
    }
    // Zero importance update accumulators (we accumulate alpha*w into importance).
    for (int i = tid; i < valid_count; i += nthrs) importance[i] = 0.0f;
    for (int b = tid; b < HIST_BINS; b += nthrs) hist[b] = 0;
    __syncthreads();

    // Tiled attention loop over all valid cache entries.
    for (int tile_start = 0; tile_start < valid_count; tile_start += FTILE) {
        int tile_len = min(FTILE, valid_count - tile_start);

        // Load K tile: [tile_len, head_dim] float16 -> float32.
        for (int i = tid; i < tile_len * head_dim; i += nthrs) {
            int p = i / head_dim, d = i % head_dim;
            K_tile[p * head_dim + d] = __half2float(cache_K[(tile_start + p) * head_dim + d]);
        }
        // Load V tile.
        for (int i = tid; i < tile_len * head_dim; i += nthrs) {
            int p = i / head_dim, d = i % head_dim;
            V_tile[p * head_dim + d] = __half2float(cache_V[(tile_start + p) * head_dim + d]);
        }
        __syncthreads();

        // For each head, compute QK^T scores and update online softmax + output.
        for (int h = 0; h < num_heads; ++h) {
            const __half* q_head = Q + h * head_dim;

            // Load q into local registers (each thread holds one dim).
            float q_val = (tid < head_dim) ? __half2float(q_head[tid]) : 0.0f;

            // Compute dot products for each tile position.
            float tile_scores[FTILE];
            for (int p = 0; p < tile_len; ++p) {
                float dot = 0.0f;
                if (tid < head_dim) {
                    dot = q_val * K_tile[p * head_dim + tid];
                }
                // Warp-level reduction across head_dim.
                for (int off = 16; off > 0; off >>= 1)
                    dot += __shfl_down_sync(0xFFFFFFFFu, dot, off);
                // Lane 0 broadcasts the full dot product.
                dot = __shfl_sync(0xFFFFFFFFu, dot, 0);
                tile_scores[p] = dot * scale;
            }

            // Online softmax update.
            float tile_max = -FLT_MAX;
            for (int p = 0; p < tile_len; ++p) tile_max = fmaxf(tile_max, tile_scores[p]);

            float old_max = max_buf[h];
            float new_max = fmaxf(old_max, tile_max);
            float scale_old = expf(old_max - new_max);

            float tile_exp[FTILE];
            float tile_sum = 0.0f;
            for (int p = 0; p < tile_len; ++p) {
                tile_exp[p] = expf(tile_scores[p] - new_max);
                tile_sum += tile_exp[p];
            }

            float new_sum = sum_buf[h] * scale_old + tile_sum;

            if (tid < head_dim) {
                // Rescale existing accumulator.
                acc_buf[h * head_dim + tid] *= scale_old;
                // Add weighted V contribution.
                float v_contrib = 0.0f;
                for (int p = 0; p < tile_len; ++p) {
                    v_contrib += tile_exp[p] * V_tile[p * head_dim + tid];
                }
                acc_buf[h * head_dim + tid] += v_contrib;
            }
            max_buf[h] = new_max;
            sum_buf[h] = new_sum;

            // Importance update: accumulate alpha * normalized_attn_weight.
            // We accumulate raw exp values and normalize after all tiles.
            // Use w_tile as scratch: lane 0 writes, all threads then read.
            if (tid < tile_len) {
                // Each thread writes its assigned tile position's weight.
                w_tile[tid] = tile_exp[tid];
            }
            __syncthreads();
            // Accumulate importance: alpha * w / new_sum (approximate with current sum).
            for (int i = tid; i < tile_len; i += nthrs) {
                float w_norm = (new_sum > 0.0f) ? w_tile[i] / new_sum : 0.0f;
                atomicAdd(&importance[tile_start + i], alpha * w_norm);
            }
            __syncthreads();
        } // end head loop
    } // end tile loop

    // Normalize output by running softmax denominator and write output.
    for (int h = 0; h < num_heads; ++h) {
        float denom = sum_buf[h];
        if (tid < head_dim) {
            float val = (denom > 0.0f) ? acc_buf[h * head_dim + tid] / denom : 0.0f;
            output[h * head_dim + tid] = __float2half(val);
        }
    }

    // Apply EMA decay: importance[p] = importance[p] + (1-alpha) * prev_importance[p]
    // (importance[p] currently holds the sum of alpha*w contributions from this step)
    for (int p = tid; p < valid_count; p += nthrs) {
        importance[p] = importance[p] + (1.0f - alpha) * prev_importance[p];
    }
    __syncthreads();

    // Check if eviction is needed.
    bool need_evict = (valid_count >= (int)(max_capacity * evict_trigger_ratio));

    if (need_evict && valid_count > 0) {
        int target_keep = (int)(max_capacity * evict_target_fraction);
        if (target_keep >= valid_count) {
            // Nothing to evict.
        } else {
            // Compute eviction threshold via histogram.
            float threshold = compute_eviction_threshold(
                hist, importance, valid_count, target_keep);

            // Compact cache.
            int new_count = compact_kv_cache_block(
                cache_K, cache_V, importance,
                valid_count, head_dim, threshold, smem_nc);

            valid_count = new_count;
            if (tid == 0) *valid_count_ptr = new_count;
            __syncthreads();
        }
    }

    // Append new K/V entry for the current step.
    // One head contributes its K/V (we use head 0's K/V per convention;
    // for multi-head KV the caller would pass per-head caches separately).
    if (valid_count < max_capacity) {
        if (tid == 0) {
            // Copy new_K[head=0] into cache.
            for (int d = 0; d < head_dim; ++d) {
                cache_K[valid_count * head_dim + d] = new_K[d];
                cache_V[valid_count * head_dim + d] = new_V[d];
            }
            importance[valid_count] = 1.0f;
            *valid_count_ptr = valid_count + 1;
        }
    }
}

void launch_fused_attention_evict(
    const __half*   Q,
    const __half*   new_K,
    const __half*   new_V,
    __half*         output,
    BoundedKVCache* cache,
    const ModelConfig& cfg,
    float*          prev_importance,
    cudaStream_t    stream)
{
    // Save importance snapshot before kernel modifies it.
    int valid_count = cache->valid_count;
    if (valid_count > 0) {
        cudaMemcpyAsync(prev_importance, cache->importance,
                        valid_count * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    int threads = cfg.head_dim;
    size_t smem =
        2 * FTILE * cfg.head_dim * sizeof(float)  // K_tile + V_tile
        + FTILE * sizeof(float)                    // w_tile
        + cfg.num_heads * cfg.head_dim * sizeof(float) // acc_buf
        + cfg.num_heads * sizeof(float)            // max_buf
        + cfg.num_heads * sizeof(float)            // sum_buf
        + HIST_BINS * sizeof(int)                  // hist
        + sizeof(int);                             // smem_nc

    // We need a device-side BoundedKVCache pointer; pass fields directly.
    fused_attention_evict_kernel<<<1, threads, smem, stream>>>(
        Q, new_K, new_V, output,
        cache->K, cache->V, cache->importance, &cache->valid_count,
        prev_importance,
        cfg.num_heads, cfg.head_dim, cfg.max_cache_capacity,
        cfg.evict_trigger_ratio, cfg.evict_target_fraction,
        cfg.importance_ema_alpha);
}
