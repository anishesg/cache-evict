"""cache_evict: bounded-memory attention with fused KV-cache eviction."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


def _reference_attention_pytorch(
    Q: torch.Tensor,  # [num_heads, head_dim]
    K: torch.Tensor,  # [seq_len, head_dim]
    V: torch.Tensor,  # [seq_len, head_dim]
) -> torch.Tensor:    # [num_heads, head_dim]
    """Pure PyTorch reference attention. Float32 accumulation."""
    head_dim = Q.shape[-1]
    scale = head_dim ** -0.5
    # Q: [H, D], K: [S, D] -> scores: [H, S]
    Q32 = Q.float()
    K32 = K.float()
    V32 = V.float()
    scores = torch.einsum("hd,sd->hs", Q32, K32) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("hs,sd->hd", weights, V32)
    return out.to(Q.dtype)


class BoundedAttention:
    """
    Autoregressive attention with a bounded KV cache.

    Maintains K, V, and importance tensors of size [max_capacity, head_dim].
    At each step, attend() runs attention over the current cache, updates
    per-position importance scores via EMA, and evicts low-importance entries
    when capacity is exceeded before appending the new K/V pair.

    All state lives on the device of the first Q tensor passed to attend().
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_capacity: int,
        importance_ema_alpha: float = 0.1,
        evict_target_fraction: float = 0.8,
        evict_trigger_ratio: float = 1.0,
    ):
        self.num_heads            = num_heads
        self.head_dim             = head_dim
        self.max_capacity         = max_capacity
        self.alpha                = importance_ema_alpha
        self.evict_target_fraction = evict_target_fraction
        self.evict_trigger_ratio  = evict_trigger_ratio

        self._device: Optional[torch.device] = None
        self._K:          Optional[torch.Tensor] = None
        self._V:          Optional[torch.Tensor] = None
        self._importance: Optional[torch.Tensor] = None
        self._valid:      int = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "BoundedAttention":
        return cls(
            num_heads             = cfg["num_heads"],
            head_dim              = cfg["head_dim"],
            max_capacity          = cfg["max_capacity"],
            importance_ema_alpha  = cfg.get("importance_ema_alpha", 0.1),
            evict_target_fraction = cfg.get("evict_target_fraction", 0.8),
            evict_trigger_ratio   = cfg.get("evict_trigger_ratio", 1.0),
        )

    def reset(self):
        """Clear cache, resetting to zero entries."""
        if self._K is not None:
            self._K.zero_()
            self._V.zero_()
            self._importance.zero_()
        self._valid = 0

    def _init_buffers(self, device: torch.device):
        self._device     = device
        self._K          = torch.zeros(self.max_capacity, self.head_dim,
                                       dtype=torch.float16, device=device)
        self._V          = torch.zeros(self.max_capacity, self.head_dim,
                                       dtype=torch.float16, device=device)
        self._importance = torch.zeros(self.max_capacity,
                                       dtype=torch.float32, device=device)

    def _try_load_extension(self):
        try:
            from cache_evict import _C  # noqa: F401
            return True
        except ImportError:
            return False

    def attend(
        self,
        q: torch.Tensor,      # [num_heads, head_dim] float16
        new_k: torch.Tensor,  # [head_dim] or [num_heads, head_dim] float16
        new_v: torch.Tensor,  # [head_dim] or [num_heads, head_dim] float16
    ) -> torch.Tensor:        # [num_heads, head_dim] float16
        """
        Run attention over the current cache and append new_k/new_v.
        Returns the attention output for the current query.
        """
        if q.dtype != torch.float16:
            raise TypeError("q must be float16")
        if q.shape != (self.num_heads, self.head_dim):
            raise ValueError(f"q shape must be [{self.num_heads}, {self.head_dim}]")

        if self._K is None:
            self._init_buffers(q.device)

        if self._valid == 0:
            # Empty cache: output is zeros, just append.
            out = torch.zeros(self.num_heads, self.head_dim,
                              dtype=torch.float16, device=q.device)
            self._append(new_k, new_v)
            return out

        # Attend over valid cache entries.
        K_valid = self._K[:self._valid]  # [valid, head_dim]
        V_valid = self._V[:self._valid]  # [valid, head_dim]
        out = _reference_attention_pytorch(q, K_valid, V_valid)

        # Compute attention weights for importance update.
        scale = self.head_dim ** -0.5
        scores = torch.einsum("hd,sd->hs", q.float(), K_valid.float()) * scale
        weights = torch.softmax(scores, dim=-1)  # [num_heads, valid]

        # EMA importance update: average across heads.
        mean_weights = weights.mean(dim=0)  # [valid]
        imp_valid = self._importance[:self._valid]
        imp_valid.mul_(1.0 - self.alpha).add_(mean_weights.to(torch.float32) * self.alpha)

        # Eviction check.
        target_keep = int(self.max_capacity * self.evict_target_fraction)
        trigger_count = int(self.max_capacity * self.evict_trigger_ratio)
        if self._valid >= trigger_count:
            self._evict(target_keep)

        self._append(new_k, new_v)
        return out

    def reference_attend(
        self,
        q: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> torch.Tensor:
        """Attend without eviction, using the full cache as oracle."""
        if self._valid == 0:
            self._init_buffers(q.device)
            out = torch.zeros(self.num_heads, self.head_dim,
                              dtype=torch.float16, device=q.device)
            self._append(new_k, new_v)
            return out

        K_valid = self._K[:self._valid]
        V_valid = self._V[:self._valid]
        out = _reference_attention_pytorch(q, K_valid, V_valid)
        self._append(new_k, new_v)
        return out

    def _evict(self, target_keep: int):
        if target_keep >= self._valid:
            return
        imp = self._importance[:self._valid]
        # Find threshold: kth-smallest importance value.
        n_evict = self._valid - target_keep
        threshold, _ = torch.kthvalue(imp, n_evict)
        keep_mask = imp > threshold.item()
        # Ensure we keep exactly target_keep (handle ties by keeping highest indices).
        keep_indices = torch.where(keep_mask)[0]
        if keep_indices.shape[0] < target_keep:
            all_indices = torch.arange(self._valid, device=self._device)
            tie_indices = torch.where(imp == threshold.item())[0]
            need = target_keep - keep_indices.shape[0]
            extra = tie_indices[:need]
            keep_indices = torch.cat([keep_indices, extra])
            keep_indices, _ = keep_indices.sort()

        n_keep = keep_indices.shape[0]
        self._K[:n_keep] = self._K[keep_indices]
        self._V[:n_keep] = self._V[keep_indices]
        self._importance[:n_keep] = self._importance[keep_indices]
        # Zero out evicted slots.
        self._K[n_keep:self._valid].zero_()
        self._V[n_keep:self._valid].zero_()
        self._importance[n_keep:self._valid].zero_()
        self._valid = n_keep

    def _append(self, new_k: torch.Tensor, new_v: torch.Tensor):
        if self._valid >= self.max_capacity:
            # Emergency FIFO eviction: drop oldest entry.
            self._K[:self.max_capacity - 1] = self._K[1:self.max_capacity].clone()
            self._V[:self.max_capacity - 1] = self._V[1:self.max_capacity].clone()
            self._importance[:self.max_capacity - 1] = self._importance[1:self.max_capacity].clone()
            self._valid = self.max_capacity - 1

        pos = self._valid
        k = new_k.to(torch.float16).reshape(self.head_dim) if new_k.dim() > 1 else new_k
        v = new_v.to(torch.float16).reshape(self.head_dim) if new_v.dim() > 1 else new_v
        self._K[pos] = k
        self._V[pos] = v
        self._importance[pos] = 1.0
        self._valid += 1

    @property
    def cache_size(self) -> int:
        return self._valid

    @property
    def K(self) -> Optional[torch.Tensor]:
        return self._K[:self._valid] if self._valid > 0 else None

    @property
    def V(self) -> Optional[torch.Tensor]:
        return self._V[:self._valid] if self._valid > 0 else None
