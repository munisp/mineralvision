"""Real PyTorch I-JEPA core for MineralVision.

Implements the binding interface contract from JEPA_REBUILD_SPEC.md:

- ViT-tiny context encoder + EMA target encoder (separate weights)
- Narrow predictor transformer
- Conv patch embedding + learned positional embeddings
- Real block masking via :func:`make_masks`
- ``train_step`` performs a full optimisation step (AdamW + EMA update)
- Pure numpy at the API surface; torch tensors are internal only.

The module imports cleanly without torch installed: ``TORCH_AVAILABLE`` is
``False``, ``jepa_backend()`` returns ``"unavailable"`` and ``JEPAModel``
raises an informative ``RuntimeError``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

try:  # import guard — module must import fine without torch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised via monkeypatched import
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False

__all__ = [
    "TORCH_AVAILABLE",
    "jepa_backend",
    "JEPAConfig",
    "JEPAModel",
    "make_masks",
]


def jepa_backend() -> str:
    """Return the active JEPA backend: ``"torch"`` or ``"unavailable"``."""
    return "torch" if TORCH_AVAILABLE else "unavailable"


@dataclass
class JEPAConfig:
    img_size: int = 96
    patch: int = 16
    embed_dim: int = 384
    depth: int = 6
    heads: int = 6
    pred_depth: int = 3
    ema_momentum: float = 0.996
    lr: float = 3e-4
    weight_decay: float = 0.05

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch) ** 2


def make_masks(
    batch: int,
    num_patches: int,
    ctx_scale: Tuple[float, float] = (0.15, 0.2),
    tgt_scale: Tuple[float, float] = (0.85, 1.0),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample real block masks for I-JEPA.

    Returns ``(ctx_idx, tgt_idx)`` as int64 arrays of shape ``[B, Nc]`` and
    ``[B, Nt]``. Target blocks are large contiguous blocks; context blocks are
    smaller blocks disjoint from the target. Counts are uniform across the
    batch so results can be batched as tensors.
    """
    rng = rng if rng is not None else np.random.default_rng()
    grid = int(round(num_patches ** 0.5))
    if grid * grid != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a perfect square")

    def sample_block(scale: Tuple[float, float], min_side: int = 1) -> list:
        area = rng.uniform(*scale) * num_patches
        aspect = float(np.exp(rng.uniform(np.log(0.75), np.log(1.5))))
        h = int(round((area * aspect) ** 0.5))
        w = int(round((area / aspect) ** 0.5))
        h = max(min_side, min(grid, h))
        w = max(min_side, min(grid, w))
        top = int(rng.integers(0, grid - h + 1))
        left = int(rng.integers(0, grid - w + 1))
        return [(top + i) * grid + (left + j) for i in range(h) for j in range(w)]

    ctx_count = max(1, int(round(rng.uniform(*ctx_scale) * num_patches)))
    max_tgt = num_patches - ctx_count  # keep enough patches for the context
    tgt_count = min(max(1, int(round(rng.uniform(*tgt_scale) * num_patches))), max_tgt)

    tgt_list = []
    for _ in range(batch):
        idxs = []
        for _attempt in range(200):
            cand = sorted(set(sample_block(tgt_scale)))
            if len(cand) >= tgt_count:
                pick = rng.choice(len(cand), size=tgt_count, replace=False)
                idxs = sorted(cand[int(c)] for c in pick)
                break
        else:  # fallback: random distinct indices
            idxs = sorted(rng.choice(num_patches, size=tgt_count, replace=False).tolist())
        tgt_list.append(idxs)

    ctx_list = []
    for tgt in tgt_list:
        remaining = [i for i in range(num_patches) if i not in set(tgt)]
        block = [i for i in sample_block(ctx_scale) if i in set(remaining)]
        # uniform count across batch: subsample or top up from remaining
        idxs = list(dict.fromkeys(block))
        if len(idxs) > ctx_count:
            pick = rng.choice(len(idxs), size=ctx_count, replace=False)
            idxs = [idxs[int(c)] for c in pick]
        elif len(idxs) < ctx_count:
            pool = [i for i in remaining if i not in set(idxs)]
            need = min(ctx_count - len(idxs), len(pool))
            if need:
                pick = rng.choice(len(pool), size=need, replace=False)
                idxs += [pool[int(e)] for e in pick]
        ctx_list.append(sorted(idxs))

    return (
        np.asarray(ctx_list, dtype=np.int64),
        np.asarray(tgt_list, dtype=np.int64),
    )


if TORCH_AVAILABLE:

    class _PatchEmbed(nn.Module):
        def __init__(self, patch: int, embed_dim: int):
            super().__init__()
            self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch, stride=patch)

        def forward(self, x):  # [B,3,H,W] -> [B,N,D]
            return self.proj(x).flatten(2).transpose(1, 2)

    class _Block(nn.Module):
        def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
            )

        def forward(self, x):
            h = self.norm1(x)
            x = x + self.attn(h, h, h, need_weights=False)[0]
            x = x + self.mlp(self.norm2(x))
            return x

    class _Encoder(nn.Module):
        """ViT-tiny encoder with learned positional embeddings."""

        def __init__(self, cfg: JEPAConfig):
            super().__init__()
            self.patch_embed = _PatchEmbed(cfg.patch, cfg.embed_dim)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, cfg.num_patches, cfg.embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.blocks = nn.ModuleList(
                [_Block(cfg.embed_dim, cfg.heads) for _ in range(cfg.depth)]
            )
            self.norm = nn.LayerNorm(cfg.embed_dim)

        def forward_tokens(self, tokens):
            for blk in self.blocks:
                tokens = blk(tokens)
            return self.norm(tokens)

        def forward(self, x, idx: Optional["torch.Tensor"] = None):
            """Encode patches; if ``idx`` given, only those positions."""
            tok = self.patch_embed(x)
            if idx is not None:
                tok = torch.gather(
                    tok, 1, idx.unsqueeze(-1).expand(-1, -1, tok.shape[-1])
                )
                tok = tok + torch.stack([self.pos_embed[0, row] for row in idx])
            else:
                tok = tok + self.pos_embed
            return self.forward_tokens(tok)

    class _Predictor(nn.Module):
        """Narrow predictor: mask tokens + context tokens -> target predictions."""

        def __init__(self, cfg: JEPAConfig):
            super().__init__()
            d = cfg.embed_dim
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, d))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.blocks = nn.ModuleList(
                [_Block(d, cfg.heads) for _ in range(cfg.pred_depth)]
            )
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, d)

        def forward(self, ctx_tokens, ctx_idx, tgt_idx):
            b, nt = tgt_idx.shape
            pos = torch.stack([self.pos_embed[0, row] for row in torch.cat([ctx_idx, tgt_idx], dim=1)])
            mask = self.mask_token.expand(b, nt, -1)
            tokens = torch.cat([ctx_tokens, mask], dim=1) + pos
            for blk in self.blocks:
                tokens = blk(tokens)
            tokens = self.norm(tokens)
            return self.head(tokens[:, ctx_idx.shape[1]:])


class JEPAModel:
    """Real I-JEPA model. Numpy in / numpy out at the API surface."""

    def __init__(self, config: Optional[JEPAConfig] = None, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "JEPAModel requires PyTorch, but torch is not installed "
                "(jepa_backend() == 'unavailable'). Install the CPU wheel: "
                "pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        self.config = config or JEPAConfig()
        self.device = torch.device(device)
        cfg = self.config
        self.context_encoder = _Encoder(cfg).to(self.device)
        self.target_encoder = _Encoder(cfg).to(self.device)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = _Predictor(cfg).to(self.device)
        params = list(self.context_encoder.parameters()) + list(self.predictor.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------ utils
    def _to_tensor(self, imgs) -> "torch.Tensor":
        arr = np.asarray(imgs, dtype=np.float32)
        if arr.ndim == 3:  # single [H,W,3]
            arr = arr[None]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"expected imgs [B,H,W,3] (or [H,W,3]), got {arr.shape}")
        if arr.shape[1] != self.config.img_size or arr.shape[2] != self.config.img_size:
            raise ValueError(
                f"expected img_size={self.config.img_size}, got {arr.shape[1]}x{arr.shape[2]}"
            )
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2)
        return t.to(self.device)

    # ------------------------------------------------------------------- API
    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def encode_target(self, imgs) -> np.ndarray:
        """Encode with the EMA target encoder -> [B, N, D] numpy (no grad)."""
        x = self._to_tensor(imgs)
        self.target_encoder.eval()
        out = self.target_encoder(x)
        return out.cpu().numpy().astype(np.float32)

    def train_step(self, imgs) -> float:
        """One real I-JEPA optimisation step. Returns the loss as a float."""
        x = self._to_tensor(imgs)
        b = x.shape[0]
        n = self.config.num_patches
        ctx_idx_np, tgt_idx_np = make_masks(b, n, rng=self.rng)
        ctx_idx = torch.from_numpy(ctx_idx_np).to(self.device)
        tgt_idx = torch.from_numpy(tgt_idx_np).to(self.device)

        with torch.no_grad():
            self.target_encoder.eval()
            tgt_all = self.target_encoder(x)  # [B,N,D]
            tgt = torch.gather(
                tgt_all, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, tgt_all.shape[-1])
            )

        self.context_encoder.train()
        self.predictor.train()
        ctx_tokens = self.context_encoder(x, idx=ctx_idx)
        pred = self.predictor(ctx_tokens, ctx_idx, tgt_idx)
        loss = F.smooth_l1_loss(pred, tgt)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # EMA update of the target encoder (no grad through target).
        m = self.config.ema_momentum
        with torch.no_grad():
            for tp, cp in zip(
                self.target_encoder.parameters(), self.context_encoder.parameters()
            ):
                tp.mul_(m).add_(cp.detach(), alpha=1.0 - m)

        return float(loss.item())

    def embed_image(self, img) -> np.ndarray:
        """Mean-pooled, L2-normalised target-encoder embedding -> [D] numpy."""
        single = np.asarray(img).ndim == 3
        tokens = self.encode_target(img)  # [B,N,D]
        emb = tokens.mean(axis=1)
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        emb = emb / np.maximum(norm, 1e-12)
        return emb[0].astype(np.float32) if single else emb.astype(np.float32)

    def save(self, path) -> None:
        """Save state_dicts + config via torch.save."""
        torch.save(
            {
                "config": asdict(self.config),
                "context_encoder": self.context_encoder.state_dict(),
                "target_encoder": self.target_encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            str(path),
        )

    @classmethod
    def load(cls, path, device: str = "cpu") -> "JEPAModel":
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "JEPAModel.load requires PyTorch, but torch is not installed."
            )
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        model = cls(JEPAConfig(**ckpt["config"]), device=device)
        model.context_encoder.load_state_dict(ckpt["context_encoder"])
        model.target_encoder.load_state_dict(ckpt["target_encoder"])
        model.predictor.load_state_dict(ckpt["predictor"])
        model.optimizer.load_state_dict(ckpt["optimizer"])
        return model
