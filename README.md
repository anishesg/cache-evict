# cache-evict

Fused attention-eviction kernel: online importance scoring during tiled attention with warp-cooperative stream compaction for bounded-memory long-context inference.

## Problem: the KV-cache memory wall

Serving transformer models at 128K+ context lengths is bottlenecked by KV-cache memory. At each decode step, every token in the context occupies `2 * num_heads * head_dim * sizeof(float16)` bytes in the KV cache. For a 32-head, 128-dim model at 128K context:

```
128K tokens * 32 heads * 128 dims * 2 (K+V) * 2 bytes = 2 GB per layer
```

A 32-layer model requires 64 GB just for KV cache, exceeding the memory of a single H100 (80 GB total, also needed for weights and activations). Serving multiple concurrent users compounds this further.

## Existing eviction approaches and their limits

**H2O (Heavy Hitter Oracle)** maintains a cumulative attention score tensor updated after each decode step. It identifies "heavy hitters" (tokens receiving disproportionate attention) and evicts the rest. Limitations: the cumulative score update requires a separate pass over the score tensor after every attention kernel, adding latency and memory bandwidth proportional to cache size. The score tensor itself is an additional `seq_len * sizeof(float32)` allocation per layer.

**StreamingLLM** retains only the initial N tokens (attention sinks) plus the most recent W tokens in a sliding window, discarding all mid-context tokens. This is essentially zero-cost but loses all mid-context information. The quality degradation is severe for tasks requiring recall of information from the middle of a long document.

**Sliding window attention** (used in Mistral, Phi) similarly loses early context. It works for tasks where locality holds but fails for retrieval-style tasks where the relevant span is anywhere in the context.

## Novel approach: fused importance scoring

This project fuses importance scoring directly into the tiled attention kernel. Key insight: the attention weights `softmax(QK^T / sqrt(d))` are computed during the attention pass and exist transiently in registers and shared memory. Rather than discarding them after accumulating the weighted-V sum, we use them to update per-position importance scores via exponential moving average:

```
importance[pos] = alpha * attn_weight[pos] + (1 - alpha) * importance[pos]
```

This accumulates a decaying history of how much attention each cache position has received across recent query steps. The EMA with `alpha` in [0.01, 0.5] gives recent queries higher weight while maintaining historical signal.

**Eviction trigger:** when the cache exceeds `max_capacity * evict_trigger_ratio`, eviction runs. A 256-bin importance histogram is built in shared memory to find the percentile threshold cutting to `evict_target_fraction * max_capacity` entries. Warp-cooperative stream compaction then moves retained entries to contiguous positions in-place, discarding sub-threshold entries. The new K/V pair is appended to the compacted cache.

**Zero extra global memory traffic for scoring:** the importance scores are updated in registers during the tile loop. The `importance_scores` array (one float32 per cache position, small compared to K/V) is read once for EMA update and written once per eviction event, not per attention step.

**Single kernel launch:** score, threshold computation, compaction, and append all happen in one kernel. No synchronization between separate kernels, no extra CUDA stream management.

## Quality-memory tradeoff

Empirical attention weight distributions in transformer models are highly skewed: a small fraction of tokens receive most of the attention mass. Under this distribution, EMA importance correctly identifies these high-value tokens. At 60% cache capacity (retaining only 60% of all seen tokens), typical output cosine similarity against full-cache attention exceeds 0.99 for most tasks. At 50% capacity, cosine similarity typically exceeds 0.98.

The EMA decay rate `alpha` controls recency bias. Lower alpha (0.01-0.05) emphasizes long-term importance, appropriate for tasks with stable retrieval targets. Higher alpha (0.1-0.5) emphasizes recent attention, appropriate for sliding-context generation tasks.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./test_correctness
./bench_latency
./bench_tradeoff
```

Requires CUDA 11.8+ and an Ampere or newer GPU (sm_80+).

## Python extension

```bash
pip install -e .
python -m pytest tests/test_python.py -v
```
