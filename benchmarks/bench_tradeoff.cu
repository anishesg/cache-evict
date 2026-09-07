#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

#include "reference_attn.cuh"
#include "fused_attn_evict.cuh"
#include "config.cuh"

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

static void rand_fp16(std::vector<__half>& v, float scale = 0.1f) {
    for (auto& x : v)
        x = __float2half(((float)rand()/(float)RAND_MAX - 0.5f)*2.0f*scale);
}

static float cosine_sim(const __half* a, const __half* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        float ai = __half2float(a[i]), bi = __half2float(b[i]);
        dot += ai*bi; na += ai*ai; nb += bi*bi;
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / sqrt(na*nb));
}

struct TradeoffResult {
    float cap_frac;
    float alpha;
    float avg_cosine;
    float worst_cosine;
    float avg_evicted_per_step;
    float peak_mb;
};

static TradeoffResult run_sweep(
    int num_heads, int head_dim, int seq_len, int total_steps,
    float cap_frac, float alpha,
    const std::vector<__half>& h_Q_all,
    const std::vector<__half>& h_K_all,
    const std::vector<__half>& h_V_all,
    __half* d_Q_all, __half* d_K_all, __half* d_V_all)
{
    int capacity = (int)(seq_len * cap_frac);
    if (capacity < 2) capacity = 2;

    ModelConfig cfg;
    cfg.num_heads             = num_heads;
    cfg.head_dim              = head_dim;
    cfg.max_cache_capacity    = capacity + 16;
    cfg.evict_target_fraction = 0.9f * cap_frac;
    cfg.importance_ema_alpha  = alpha;
    cfg.evict_trigger_ratio   = 1.0f;

    BoundedKVCache cache = alloc_cache(cfg.max_cache_capacity, head_dim);
    float* d_prev_imp;
    CUDA_CHECK(cudaMalloc(&d_prev_imp, cfg.max_cache_capacity * sizeof(float)));

    __half *d_out_fused, *d_out_ref;
    CUDA_CHECK(cudaMalloc(&d_out_fused, num_heads * head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out_ref,   num_heads * head_dim * sizeof(__half)));

    std::vector<__half> h_out_fused(num_heads * head_dim);
    std::vector<__half> h_out_ref(num_heads * head_dim);

    double sum_cosine    = 0.0;
    float  worst_cosine  = 1.0f;
    long   total_evicted = 0;
    int    meas_steps    = 0;
    int    prev_valid    = 0;

    for (int step = 0; step < total_steps; ++step) {
        int seq_idx = step % total_steps;
        __half* q = d_Q_all + seq_idx * num_heads * head_dim;
        __half* k = d_K_all + seq_idx * head_dim;
        __half* v = d_V_all + seq_idx * head_dim;

        // Get current valid_count before this step to measure evictions.
        int cur_valid;
        CUDA_CHECK(cudaMemcpy(&cur_valid, &cache.valid_count, sizeof(int), cudaMemcpyDeviceToHost));

        launch_fused_attention_evict(q, k, v, d_out_fused, &cache, cfg, d_prev_imp);
        CUDA_CHECK(cudaDeviceSynchronize());

        int new_valid;
        CUDA_CHECK(cudaMemcpy(&new_valid, &cache.valid_count, sizeof(int), cudaMemcpyDeviceToHost));
        int evicted_this_step = (cur_valid + 1) - new_valid; // +1 for the appended entry
        if (evicted_this_step < 0) evicted_this_step = 0;
        total_evicted += evicted_this_step;

        // After warmup, measure quality.
        if (step >= seq_len / 2) {
            // Reference: full-cache attention over all tokens seen so far.
            int ref_len = std::min(step + 1, seq_len);
            launch_reference_attention(q, d_K_all, d_V_all, d_out_ref,
                                       num_heads, head_dim, ref_len);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_out_fused.data(), d_out_fused, h_out_fused.size()*sizeof(__half), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_out_ref.data(),   d_out_ref,   h_out_ref.size()*sizeof(__half),   cudaMemcpyDeviceToHost));

            float cs = cosine_sim(h_out_fused.data(), h_out_ref.data(), num_heads * head_dim);
            sum_cosine += cs;
            if (cs < worst_cosine) worst_cosine = cs;
            ++meas_steps;
        }
        prev_valid = new_valid;
    }

    float peak_mb = (float)cfg.max_cache_capacity * head_dim * 2 * sizeof(__half) / (1024.0f * 1024.0f);

    cudaFree(d_prev_imp);
    cudaFree(d_out_fused);
    cudaFree(d_out_ref);
    free_cache(cache);

    TradeoffResult r;
    r.cap_frac            = cap_frac;
    r.alpha               = alpha;
    r.avg_cosine          = meas_steps > 0 ? (float)(sum_cosine / meas_steps) : 0.0f;
    r.worst_cosine        = worst_cosine;
    r.avg_evicted_per_step = (float)total_evicted / total_steps;
    r.peak_mb             = peak_mb;
    return r;
}

int main() {
    srand(42);

    const int num_heads   = 32;
    const int head_dim    = 128;
    const int seq_len     = 8192;
    const int total_steps = seq_len;

    // Pre-generate random inputs.
    std::vector<__half> h_Q_all(total_steps * num_heads * head_dim);
    std::vector<__half> h_K_all(total_steps * head_dim);
    std::vector<__half> h_V_all(total_steps * head_dim);
    rand_fp16(h_Q_all); rand_fp16(h_K_all); rand_fp16(h_V_all);

    __half *d_Q_all, *d_K_all, *d_V_all;
    CUDA_CHECK(cudaMalloc(&d_Q_all, h_Q_all.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_K_all, h_K_all.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_V_all, h_V_all.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_Q_all, h_Q_all.data(), h_Q_all.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_all, h_K_all.data(), h_K_all.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_all, h_V_all.data(), h_V_all.size()*sizeof(__half), cudaMemcpyHostToDevice));

    // Sweep 1: capacity fraction 20% to 100% in 5% steps, fixed alpha=0.1.
    printf("=== Sweep 1: cache capacity fraction (alpha=0.1) ===\n");
    printf("%-8s  %-10s  %-12s  %-12s  %-14s  %-10s\n",
           "cap%", "peak_mb", "avg_cosine", "worst_cosine", "evict/step", "peak_mb");
    printf("%-8s  %-10s  %-12s  %-12s  %-14s  %-10s\n",
           "----", "-------", "----------", "------------", "----------", "-------");

    for (int pct = 20; pct <= 100; pct += 5) {
        float cap = pct / 100.0f;
        auto r = run_sweep(num_heads, head_dim, seq_len, total_steps,
                           cap, 0.1f,
                           h_Q_all, h_K_all, h_V_all,
                           d_Q_all, d_K_all, d_V_all);
        printf("%-8.0f  %-10.2f  %-12.6f  %-12.6f  %-14.2f  %-10.2f\n",
               cap * 100.0f, r.peak_mb, r.avg_cosine, r.worst_cosine,
               r.avg_evicted_per_step, r.peak_mb);
    }

    // Sweep 2: alpha variation at 60% capacity.
    printf("\n=== Sweep 2: EMA alpha (cap=60%%) ===\n");
    printf("%-8s  %-12s  %-12s  %-14s\n",
           "alpha", "avg_cosine", "worst_cosine", "evict/step");
    printf("%-8s  %-12s  %-12s  %-14s\n",
           "-----", "----------", "------------", "----------");

    float alphas[] = {0.01f, 0.02f, 0.05f, 0.1f, 0.15f, 0.2f, 0.3f, 0.5f};
    for (float a : alphas) {
        auto r = run_sweep(num_heads, head_dim, seq_len, total_steps,
                           0.6f, a,
                           h_Q_all, h_K_all, h_V_all,
                           d_Q_all, d_K_all, d_V_all);
        printf("%-8.3f  %-12.6f  %-12.6f  %-14.2f\n",
               a, r.avg_cosine, r.worst_cosine, r.avg_evicted_per_step);
    }

    cudaFree(d_Q_all);
    cudaFree(d_K_all);
    cudaFree(d_V_all);
    return 0;
}
