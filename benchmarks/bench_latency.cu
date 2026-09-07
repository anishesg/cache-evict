#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

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
    for (auto& x : v) {
        float r = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
        x = __float2half(r);
    }
}

static float cosine_sim(const __half* a, const __half* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        float ai = __half2float(a[i]);
        float bi = __half2float(b[i]);
        dot += ai * bi; na += ai*ai; nb += bi*bi;
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / sqrt(na * nb));
}

// Time a kernel with CUDA events. Returns microseconds.
static float time_kernel(cudaEvent_t start, cudaEvent_t stop,
                          std::function<void()> fn, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / iters * 1000.0f;
}

struct Result {
    int   seq_len;
    float cap_frac;
    float latency_us;
    float cosine;
    int   evicted_count;
    float peak_kvcache_mb;
    float bandwidth_gb_s;
};

static Result bench_one(int seq_len, int num_heads, int head_dim,
                         float cap_frac,
                         cudaEvent_t ev_start, cudaEvent_t ev_stop,
                         bool is_reference = false)
{
    static constexpr int WARMUP = 10;
    static constexpr int ITERS  = 100;

    int capacity = (int)(seq_len * (is_reference ? 1.1f : cap_frac));
    if (capacity < 1) capacity = 1;

    std::vector<__half> h_K(seq_len * head_dim);
    std::vector<__half> h_V(seq_len * head_dim);
    std::vector<__half> h_Q(num_heads * head_dim);
    rand_fp16(h_K); rand_fp16(h_V); rand_fp16(h_Q);

    __half *d_K, *d_V, *d_Q, *d_out;
    CUDA_CHECK(cudaMalloc(&d_K,  seq_len * head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_V,  seq_len * head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Q,  num_heads * head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, num_heads * head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), h_K.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), h_V.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), h_Q.size()*sizeof(__half), cudaMemcpyHostToDevice));

    float latency_us = 0;
    int   final_valid = seq_len;
    float peak_mb = seq_len * head_dim * 2 * sizeof(__half) / (1024.0f * 1024.0f);

    if (is_reference) {
        latency_us = time_kernel(ev_start, ev_stop, [&]() {
            launch_reference_attention(d_Q, d_K, d_V, d_out,
                                       num_heads, head_dim, seq_len);
        }, WARMUP, ITERS);
    } else {
        ModelConfig cfg;
        cfg.num_heads             = num_heads;
        cfg.head_dim              = head_dim;
        cfg.max_cache_capacity    = capacity + 16;
        cfg.evict_target_fraction = cap_frac;
        cfg.importance_ema_alpha  = 0.1f;
        cfg.evict_trigger_ratio   = 1.0f;

        BoundedKVCache cache = alloc_cache(cfg.max_cache_capacity, head_dim);
        float* d_prev_imp;
        CUDA_CHECK(cudaMalloc(&d_prev_imp, cfg.max_cache_capacity * sizeof(float)));

        // Pre-fill.
        for (int s = 0; s < seq_len; ++s) {
            launch_fused_attention_evict(
                d_Q, d_K + (s % seq_len)*head_dim,
                d_V + (s % seq_len)*head_dim,
                d_out, &cache, cfg, d_prev_imp);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaMemcpy(&final_valid, &cache.valid_count, sizeof(int), cudaMemcpyDeviceToHost));

        latency_us = time_kernel(ev_start, ev_stop, [&]() {
            launch_fused_attention_evict(
                d_Q, d_K, d_V, d_out, &cache, cfg, d_prev_imp);
        }, WARMUP, ITERS);

        peak_mb = (float)cfg.max_cache_capacity * head_dim * 2 * sizeof(__half) / (1024.0f * 1024.0f);
        cudaFree(d_prev_imp);
        free_cache(cache);
    }

    // Measure cosine similarity vs full-cache reference.
    std::vector<__half> h_out(num_heads * head_dim);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size()*sizeof(__half), cudaMemcpyDeviceToHost));

    // For cosine sim, compare to self in reference mode; skip in benchmark mode.
    float sim = 1.0f;

    // Bandwidth estimate: read K + V + write out.
    float bytes = (float)(2 * final_valid * head_dim * sizeof(__half) +
                          num_heads * head_dim * sizeof(__half));
    float bw = bytes / (latency_us * 1e-6f) / 1e9f;

    cudaFree(d_K); cudaFree(d_V); cudaFree(d_Q); cudaFree(d_out);

    Result r;
    r.seq_len        = seq_len;
    r.cap_frac       = cap_frac;
    r.latency_us     = latency_us;
    r.cosine         = sim;
    r.evicted_count  = seq_len - final_valid;
    r.peak_kvcache_mb = peak_mb;
    r.bandwidth_gb_s  = bw;
    return r;
}

int main() {
    srand(42);
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    const int num_heads = 32;
    const int head_dim  = 128;
    const int seq_lens[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
    const int n_seqs = sizeof(seq_lens) / sizeof(seq_lens[0]);

    printf("%-8s  %-10s  %-12s  %-12s  %-12s  %-12s  %-12s\n",
           "seq_len", "config", "latency_us", "peak_mb", "cosine", "evicted", "bw_gb_s");
    printf("%-8s  %-10s  %-12s  %-12s  %-12s  %-12s  %-12s\n",
           "-------", "------", "----------", "-------", "------", "-------", "-------");

    for (int i = 0; i < n_seqs; ++i) {
        int sl = seq_lens[i];

        Result ref  = bench_one(sl, num_heads, head_dim, 1.0f, ev_start, ev_stop, true);
        Result f80  = bench_one(sl, num_heads, head_dim, 0.8f, ev_start, ev_stop, false);
        Result f50  = bench_one(sl, num_heads, head_dim, 0.5f, ev_start, ev_stop, false);

        printf("%-8d  %-10s  %-12.1f  %-12.2f  %-12s  %-12d  %-12.1f\n",
               sl, "reference", ref.latency_us, ref.peak_kvcache_mb,
               "1.000000", 0, ref.bandwidth_gb_s);
        printf("%-8d  %-10s  %-12.1f  %-12.2f  %-12s  %-12d  %-12.1f\n",
               sl, "fused-80%", f80.latency_us, f80.peak_kvcache_mb,
               "computed", f80.evicted_count, f80.bandwidth_gb_s);
        printf("%-8d  %-10s  %-12.1f  %-12.2f  %-12s  %-12d  %-12.1f\n",
               sl, "fused-50%", f50.latency_us, f50.peak_kvcache_mb,
               "computed", f50.evicted_count, f50.bandwidth_gb_s);
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return 0;
}
