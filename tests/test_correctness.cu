#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>

#include "reference_attn.cuh"
#include "fused_attn_evict.cuh"
#include "config.cuh"

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static void rand_fp16(std::vector<__half>& v, float scale = 0.1f) {
    for (auto& x : v) {
        float r = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
        x = __float2half(r);
    }
}

// Cosine similarity between two float16 arrays.
static float cosine_sim(const __half* a, const __half* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        float ai = __half2float(a[i]);
        float bi = __half2float(b[i]);
        dot += ai * bi;
        na  += ai * ai;
        nb  += bi * bi;
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / sqrt(na * nb));
}

struct TestConfig {
    int num_heads;
    int head_dim;
    int seq_len;
};

static bool run_test(const TestConfig& tc, float capacity_frac, float min_cosine,
                     bool expect_eviction, const char* label)
{
    int num_heads = tc.num_heads;
    int head_dim  = tc.head_dim;
    int seq_len   = tc.seq_len;
    int capacity  = (int)(seq_len * capacity_frac);
    if (capacity < 1) capacity = 1;

    // Build host K/V arrays for reference.
    std::vector<__half> h_K(seq_len * head_dim);
    std::vector<__half> h_V(seq_len * head_dim);
    std::vector<__half> h_Q(num_heads * head_dim);
    rand_fp16(h_K);
    rand_fp16(h_V);
    rand_fp16(h_Q);

    // Device buffers for reference.
    __half *d_K, *d_V, *d_Q, *d_ref_out;
    CUDA_CHECK(cudaMalloc(&d_K,       seq_len   * head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_V,       seq_len   * head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Q,       num_heads * head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ref_out, num_heads * head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), h_K.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), h_V.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), h_Q.size()*sizeof(__half), cudaMemcpyHostToDevice));

    launch_reference_attention(d_Q, d_K, d_V, d_ref_out,
                                num_heads, head_dim, seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fused kernel test: load seq_len entries into a bounded cache, then query.
    ModelConfig cfg;
    cfg.num_heads            = num_heads;
    cfg.head_dim             = head_dim;
    cfg.max_cache_capacity   = capacity + num_heads; // +num_heads headroom
    cfg.evict_target_fraction = 0.8f;
    cfg.importance_ema_alpha  = 0.1f;
    cfg.evict_trigger_ratio   = (capacity_frac >= 1.0f) ? 2.0f : 1.0f; // trigger at 100%

    BoundedKVCache cache = alloc_cache(cfg.max_cache_capacity, head_dim);

    // Populate cache by simulating seq_len decode steps.
    float* d_prev_imp;
    CUDA_CHECK(cudaMalloc(&d_prev_imp, cfg.max_cache_capacity * sizeof(float)));
    __half *d_fused_out;
    CUDA_CHECK(cudaMalloc(&d_fused_out, num_heads * head_dim * sizeof(__half)));

    // Pre-fill cache with the first seq_len entries.
    // We call the fused kernel repeatedly but suppress eviction until full.
    for (int step = 0; step < seq_len; ++step) {
        __half *step_K = d_K + (step % seq_len) * head_dim;
        __half *step_V = d_V + (step % seq_len) * head_dim;
        // Use a dummy query for pre-fill; output is ignored.
        launch_fused_attention_evict(
            d_Q, step_K, step_V, d_fused_out,
            &cache, cfg, d_prev_imp);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Final query using the bounded cache.
    int final_valid;
    CUDA_CHECK(cudaMemcpy(&final_valid, &cache.valid_count, sizeof(int), cudaMemcpyDeviceToHost));

    // Copy fused output.
    std::vector<__half> h_ref_out(num_heads * head_dim);
    std::vector<__half> h_fused_out(num_heads * head_dim);
    CUDA_CHECK(cudaMemcpy(h_ref_out.data(),   d_ref_out,   h_ref_out.size()*sizeof(__half),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fused_out.data(), d_fused_out, h_fused_out.size()*sizeof(__half), cudaMemcpyDeviceToHost));

    float sim = cosine_sim(h_ref_out.data(), h_fused_out.data(), num_heads * head_dim);

    bool evicted = (final_valid <= capacity);
    bool quality_ok = (sim >= min_cosine);
    bool eviction_ok = !expect_eviction || evicted;

    printf("[%s] heads=%d dim=%d seq=%d cap=%.0f%% cosine=%.6f valid=%d/%d %s\n",
           label, num_heads, head_dim, seq_len, capacity_frac*100.0f,
           sim, final_valid, cfg.max_cache_capacity,
           (quality_ok && eviction_ok) ? "PASS" : "FAIL");

    cudaFree(d_K); cudaFree(d_V); cudaFree(d_Q);
    cudaFree(d_ref_out); cudaFree(d_fused_out); cudaFree(d_prev_imp);
    free_cache(cache);

    return quality_ok && eviction_ok;
}

int main() {
    srand(42);
    int failures = 0;

    TestConfig configs[] = {
        {8,  64,  512},
        {8,  128, 512},
        {32, 64,  512},
        {32, 128, 512},
        {8,  64,  2048},
        {8,  128, 2048},
        {32, 64,  2048},
        {32, 128, 2048},
        {8,  128, 8192},
        {32, 128, 8192},
    };

    for (auto& tc : configs) {
        // Test 1: full capacity, no eviction triggered, cosine > 0.9999.
        if (!run_test(tc, 1.1f, 0.9999f, false, "full-capacity")) ++failures;

        // Test 2: 80% capacity.
        if (!run_test(tc, 0.8f, 0.995f, true, "80pct-capacity")) ++failures;

        // Test 3: 50% capacity.
        if (!run_test(tc, 0.5f, 0.98f, true, "50pct-capacity")) ++failures;
    }

    printf("\n%s: %d failure(s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
