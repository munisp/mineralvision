# JEPA_REBUILD_SPEC.md — Real JEPA for MineralVision

## Interface contract (binding for J1 + J2)
New module `MineralVision_Final_Package/src/api/jepa/torch_core.py`:

```python
TORCH_AVAILABLE: bool                      # import-guarded
def jepa_backend() -> str                  # "torch" | "unavailable"
class JEPAConfig:                          # dataclass
    img_size: int = 96; patch: int = 16; embed_dim: int = 384
    depth: int = 6; heads: int = 6; pred_depth: int = 3
    ema_momentum: float = 0.996
class JEPAModel:
    def __init__(self, config: JEPAConfig, device: str = "cpu") ...
    def encode_target(self, imgs) -> "np.ndarray [B, N, D]"      # EMA target encoder, no grad
    def train_step(self, imgs) -> float                          # context encode (masked) -> predict -> smooth-L1 vs target -> backprop -> EMA update; returns real loss
    def embed_image(self, img) -> "np.ndarray [D]"               # mean-pooled target-encoder embedding, L2-normalized
    def save(self, path) / @classmethod load(cls, path, device)  # torch state_dict checkpoint
def make_masks(batch, num_patches, ctx_scale=(0.15,0.2), tgt_scale=(0.85,1.0), rng) -> (ctx_idx, tgt_idx)  # real block masking
```
- Deterministic with torch.manual_seed; CPU-feasible (img 96, patch 16 → 36 tokens, ViT-tiny depths above).
- J1 owns this file + its unit tests. J2 codes ONLY against this contract.

## Global contract
Same as GEOSPATIAL_SPEC: routers exported from modules, tests seeded/real, no silent mocks (unavailable torch → 503 / honest `backend` field), dual-context imports, no main.py edits (orchestrator wires), commit early/often, never push.
