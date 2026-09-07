"""Pytest suite for BoundedAttention Python API."""

import math
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cache_evict import BoundedAttention


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


NUM_HEADS = 8
HEAD_DIM  = 64
CAPACITY  = int(2048 * 0.6)  # 60% of total queries


@pytest.fixture
def rng():
    torch.manual_seed(42)
    return torch.Generator().manual_seed(42)


def make_qkv(num_heads: int, head_dim: int, rng: torch.Generator):
    q = torch.randn(num_heads, head_dim, generator=rng).half()
    k = torch.randn(head_dim,            generator=rng).half()
    v = torch.randn(head_dim,            generator=rng).half()
    return q, k, v


def test_output_cosine_similarity_60pct():
    """
    Feed 2048 queries through BoundedAttention at 60% capacity.
    Average cosine similarity vs full-cache reference must exceed 0.99.
    """
    torch.manual_seed(0)
    total = 2048
    bounded = BoundedAttention(
        num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        max_capacity=CAPACITY,
        importance_ema_alpha=0.1,
        evict_target_fraction=0.8,
        evict_trigger_ratio=1.0,
    )
    reference = BoundedAttention(
        num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        max_capacity=total + 10,  # effectively unbounded
        importance_ema_alpha=0.1,
        evict_target_fraction=1.0,
        evict_trigger_ratio=999.0,
    )

    qs = [torch.randn(NUM_HEADS, HEAD_DIM).half() for _ in range(total)]
    ks = [torch.randn(HEAD_DIM).half()            for _ in range(total)]
    vs = [torch.randn(HEAD_DIM).half()            for _ in range(total)]

    sims = []
    for i in range(total):
        out_b = bounded.attend(qs[i], ks[i], vs[i])
        out_r = reference.attend(qs[i], ks[i], vs[i])

        if i >= total // 4:  # skip initial warm-up where caches are mostly empty
            sims.append(cosine_sim(out_b, out_r))

    avg_sim = sum(sims) / len(sims)
    print(f"\navg cosine similarity (60% capacity): {avg_sim:.6f}")
    assert avg_sim > 0.99, f"expected avg cosine > 0.99, got {avg_sim:.6f}"


def test_cache_size_never_exceeds_capacity():
    """
    After the warmup period (initial fill), cache_size must never exceed max_capacity.
    """
    torch.manual_seed(1)
    total = 2048
    bounded = BoundedAttention(
        num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        max_capacity=CAPACITY,
        importance_ema_alpha=0.1,
        evict_target_fraction=0.8,
        evict_trigger_ratio=1.0,
    )

    for i in range(total):
        q = torch.randn(NUM_HEADS, HEAD_DIM).half()
        k = torch.randn(HEAD_DIM).half()
        v = torch.randn(HEAD_DIM).half()
        bounded.attend(q, k, v)
        assert bounded.cache_size <= bounded.max_capacity, (
            f"step {i}: cache_size={bounded.cache_size} exceeds max_capacity={bounded.max_capacity}"
        )


def test_reset_clears_state():
    """reset() must bring cache_size to 0."""
    torch.manual_seed(2)
    bounded = BoundedAttention(
        num_heads=NUM_HEADS, head_dim=HEAD_DIM, max_capacity=128)
    for _ in range(50):
        q = torch.randn(NUM_HEADS, HEAD_DIM).half()
        k = torch.randn(HEAD_DIM).half()
        v = torch.randn(HEAD_DIM).half()
        bounded.attend(q, k, v)
    assert bounded.cache_size > 0
    bounded.reset()
    assert bounded.cache_size == 0


def test_from_config_factory():
    """from_config() must produce a usable BoundedAttention with correct params."""
    cfg = {
        "num_heads": 4,
        "head_dim": 32,
        "max_capacity": 64,
        "importance_ema_alpha": 0.05,
        "evict_target_fraction": 0.75,
        "evict_trigger_ratio": 1.0,
    }
    ba = BoundedAttention.from_config(cfg)
    assert ba.num_heads == 4
    assert ba.head_dim == 32
    assert ba.max_capacity == 64
    assert ba.alpha == 0.05
    assert ba.evict_target_fraction == 0.75

    q = torch.randn(4, 32).half()
    k = torch.randn(32).half()
    v = torch.randn(32).half()
    out = ba.attend(q, k, v)
    assert out.shape == (4, 32)
    assert out.dtype == torch.float16


def test_dtype_rejection():
    """attend() must raise TypeError for non-float16 queries."""
    ba = BoundedAttention(num_heads=4, head_dim=32, max_capacity=64)
    q = torch.randn(4, 32)  # float32
    k = torch.randn(32).half()
    v = torch.randn(32).half()
    with pytest.raises(TypeError):
        ba.attend(q, k, v)
