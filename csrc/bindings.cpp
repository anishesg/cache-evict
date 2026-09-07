#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Forward declarations from CUDA TUs.
void launch_reference_attention(
    const __half* Q, const __half* K, const __half* V, __half* output,
    int num_heads, int head_dim, int seq_len, cudaStream_t stream);

#include "config.cuh"
void launch_fused_attention_evict(
    const __half* Q, const __half* new_K, const __half* new_V, __half* output,
    BoundedKVCache* cache, const ModelConfig& cfg, float* prev_importance,
    cudaStream_t stream);

static void check_fp16(const torch::Tensor& t, const char* name) {
    if (t.scalar_type() != torch::kFloat16)
        throw std::invalid_argument(std::string(name) + " must be float16");
    if (!t.is_contiguous())
        throw std::invalid_argument(std::string(name) + " must be contiguous");
    if (!t.is_cuda())
        throw std::invalid_argument(std::string(name) + " must be on CUDA device");
}

static void check_fp32(const torch::Tensor& t, const char* name) {
    if (t.scalar_type() != torch::kFloat32)
        throw std::invalid_argument(std::string(name) + " must be float32");
    if (!t.is_contiguous())
        throw std::invalid_argument(std::string(name) + " must be contiguous");
    if (!t.is_cuda())
        throw std::invalid_argument(std::string(name) + " must be on CUDA device");
}

// reference_attention(Q, K, V) -> output
// Q: [num_heads, head_dim] float16
// K: [seq_len, head_dim]   float16
// V: [seq_len, head_dim]   float16
torch::Tensor py_reference_attention(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    check_fp16(Q, "Q");
    check_fp16(K, "K");
    check_fp16(V, "V");

    if (Q.dim() != 2) throw std::invalid_argument("Q must be 2D [num_heads, head_dim]");
    if (K.dim() != 2) throw std::invalid_argument("K must be 2D [seq_len, head_dim]");
    if (V.dim() != 2) throw std::invalid_argument("V must be 2D [seq_len, head_dim]");
    if (K.size(1) != Q.size(1)) throw std::invalid_argument("K/Q head_dim mismatch");
    if (V.sizes() != K.sizes())  throw std::invalid_argument("K/V shape mismatch");

    int num_heads = Q.size(0);
    int head_dim  = Q.size(1);
    int seq_len   = K.size(0);

    auto output = torch::empty({num_heads, head_dim},
                               Q.options().dtype(torch::kFloat16));
    launch_reference_attention(
        reinterpret_cast<const __half*>(Q.data_ptr()),
        reinterpret_cast<const __half*>(K.data_ptr()),
        reinterpret_cast<const __half*>(V.data_ptr()),
        reinterpret_cast<__half*>(output.data_ptr()),
        num_heads, head_dim, seq_len,
        at::cuda::getCurrentCUDAStream());
    return output;
}

// fused_attention_evict(Q, new_K, new_V, cache_K, cache_V, importance_scores,
//                       valid_count_tensor, num_heads, head_dim, max_capacity,
//                       evict_trigger_ratio, evict_target_fraction, alpha)
// Returns: (output, updated_cache_K, updated_cache_V, updated_importance, valid_count_tensor)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
py_fused_attention_evict(
    torch::Tensor Q,
    torch::Tensor new_K,
    torch::Tensor new_V,
    torch::Tensor cache_K,
    torch::Tensor cache_V,
    torch::Tensor importance_scores,
    torch::Tensor valid_count_t,
    int    max_capacity,
    float  evict_trigger_ratio,
    float  evict_target_fraction,
    float  alpha)
{
    check_fp16(Q, "Q");
    check_fp16(new_K, "new_K");
    check_fp16(new_V, "new_V");
    check_fp16(cache_K, "cache_K");
    check_fp16(cache_V, "cache_V");
    check_fp32(importance_scores, "importance_scores");

    int num_heads = Q.size(0);
    int head_dim  = Q.size(1);

    if (valid_count_t.scalar_type() != torch::kInt32 || !valid_count_t.is_cuda())
        throw std::invalid_argument("valid_count must be int32 on CUDA");

    auto output = torch::empty({num_heads, head_dim}, Q.options());

    ModelConfig cfg;
    cfg.num_heads             = num_heads;
    cfg.head_dim              = head_dim;
    cfg.max_cache_capacity    = max_capacity;
    cfg.evict_trigger_ratio   = evict_trigger_ratio;
    cfg.evict_target_fraction = evict_target_fraction;
    cfg.importance_ema_alpha  = alpha;

    // Build a BoundedKVCache from provided tensors.
    BoundedKVCache cache;
    cache.K          = reinterpret_cast<__half*>(cache_K.data_ptr());
    cache.V          = reinterpret_cast<__half*>(cache_V.data_ptr());
    cache.importance = importance_scores.data_ptr<float>();
    cache.max_capacity = max_capacity;
    // valid_count lives in device memory; we pass its pointer.
    cache.valid_count = valid_count_t.item<int>();

    auto prev_imp = torch::empty({max_capacity},
                                 importance_scores.options());

    // We need device-side valid_count. Patch cache.valid_count field
    // by copying from device, running kernel with host-side struct, then
    // writing back. The kernel updates valid_count via pointer.
    int* d_valid_count = valid_count_t.data_ptr<int>();

    // Temporarily copy the host valid_count into the struct.
    // The kernel takes &cache.valid_count as a device pointer, but our
    // BoundedKVCache is host-side. We use a device int from valid_count_t.
    // Re-wire: pass d_valid_count directly to the kernel.
    // We call the kernel directly rather than through launch_fused_attention_evict
    // to avoid the extra memcpy of valid_count.

    // Simple path: use launch wrapper, then update valid_count_t.
    TORCH_CHECK(false, "Use BoundedAttention.attend() from Python; direct binding not yet wired for device valid_count");

    return {output, cache_K, cache_V, importance_scores, valid_count_t};
}

// compact_cache(cache_K, cache_V, importance, valid_count, threshold) -> new_valid_count
// Runs stream compaction on CPU for testing purposes.
// CUDA-side compaction is invoked through the fused kernel.
int py_compact_cache_count(
    torch::Tensor importance,
    int           valid_count,
    float         threshold)
{
    check_fp32(importance, "importance");
    auto imp_cpu = importance.cpu();
    float* imp_ptr = imp_cpu.data_ptr<float>();
    int new_count = 0;
    for (int i = 0; i < valid_count; ++i) {
        if (imp_ptr[i] > threshold) ++new_count;
    }
    return new_count;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cache-evict: fused attention-eviction CUDA kernels";

    m.def("reference_attention", &py_reference_attention,
          "Dense tiled attention reference (no eviction). Q/K/V float16.",
          py::arg("Q"), py::arg("K"), py::arg("V"));

    m.def("compact_cache_count", &py_compact_cache_count,
          "Count retained entries after compaction at given threshold.",
          py::arg("importance"), py::arg("valid_count"), py::arg("threshold"));
}
