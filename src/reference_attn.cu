#include "reference_attn.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

static constexpr int TILE_SEQ = 64;

// One block per (head). Threads cooperate on a single head's attention.
// blockDim.x = head_dim (up to 128), one block per head.
__global__ void reference_attention_kernel(
    const __half* __restrict__ Q,       // [num_heads, head_dim]
    const __half* __restrict__ K,       // [seq_len, head_dim]
    const __half* __restrict__ V,       // [seq_len, head_dim]
    __half*       __restrict__ output,  // [num_heads, head_dim]
    int head_dim,
    int seq_len)
{
    int head = blockIdx.x;
    int tid  = threadIdx.x;  // tid < head_dim

    extern __shared__ float smem[];
    float* K_tile = smem;                     // [TILE_SEQ, head_dim]
    float* V_tile = smem + TILE_SEQ * head_dim; // [TILE_SEQ, head_dim]

    const __half* q = Q + head * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Load query into registers.
    float q_val = (tid < head_dim) ? __half2float(q[tid]) : 0.0f;

    // Online softmax state.
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float acc = 0.0f;  // output accumulator for this thread's dimension

    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SEQ) {
        int tile_len = min(TILE_SEQ, seq_len - tile_start);

        // Cooperative load of K tile and V tile into shared memory.
        for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
            int pos = i / head_dim;
            int dim = i % head_dim;
            K_tile[pos * head_dim + dim] = __half2float(K[(tile_start + pos) * head_dim + dim]);
            V_tile[pos * head_dim + dim] = __half2float(V[(tile_start + pos) * head_dim + dim]);
        }
        __syncthreads();

        // Compute QK^T dot products for this tile. Each thread computes all
        // positions but only uses its own dim for reduction via warp shuffle.
        float scores[TILE_SEQ];
        for (int p = 0; p < tile_len; ++p) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d += blockDim.x) {
                int dim = d + tid;
                if (dim < head_dim) {
                    dot += q_val * K_tile[p * head_dim + tid];
                }
            }
            // Reduce dot product across threads in the block.
            // Using shared memory partial sums for simplicity.
            // We reuse K_tile[0..head_dim] as scratch for this reduction.
            K_tile[tid] = (tid < head_dim) ? q_val * K_tile[p * head_dim + tid] : 0.0f;
            __syncthreads();
            // Parallel reduction in smem.
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) K_tile[tid] += K_tile[tid + stride];
                __syncthreads();
            }
            scores[p] = K_tile[0] * scale;
            // Restore K_tile for the current position (already consumed).
            __syncthreads();
        }

        // Update online softmax: find tile max.
        float tile_max = -FLT_MAX;
        for (int p = 0; p < tile_len; ++p) tile_max = fmaxf(tile_max, scores[p]);

        float new_max = fmaxf(running_max, tile_max);
        float exp_scale_old = expf(running_max - new_max);
        float tile_sum = 0.0f;
        float tile_scores_exp[TILE_SEQ];
        for (int p = 0; p < tile_len; ++p) {
            tile_scores_exp[p] = expf(scores[p] - new_max);
            tile_sum += tile_scores_exp[p];
        }

        // Rescale accumulator and add weighted V.
        acc *= exp_scale_old;
        running_sum = running_sum * exp_scale_old + tile_sum;
        running_max = new_max;

        if (tid < head_dim) {
            float v_contrib = 0.0f;
            for (int p = 0; p < tile_len; ++p) {
                v_contrib += tile_scores_exp[p] * V_tile[p * head_dim + tid];
            }
            acc += v_contrib;
        }
        __syncthreads();
    }

    // Normalize and write output.
    if (tid < head_dim && running_sum > 0.0f) {
        output[head * head_dim + tid] = __float2half(acc / running_sum);
    }
}

void launch_reference_attention(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       output,
    int           num_heads,
    int           head_dim,
    int           seq_len,
    cudaStream_t  stream)
{
    int threads = head_dim;
    // Shared memory: K tile + V tile, each TILE_SEQ * head_dim floats.
    size_t smem = 2 * TILE_SEQ * head_dim * sizeof(float);
    reference_attention_kernel<<<num_heads, threads, smem, stream>>>(
        Q, K, V, output, head_dim, seq_len);
}
