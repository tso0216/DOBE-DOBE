"""Model loading and deterministic latent inference for the web dashboard.

The training folders are scripts rather than importable Python packages.  This
module mirrors only their inference-time architectures so the website can load
several checkpoints in the same process without collisions between the many
``cfg.py``/``model.py`` module names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_CATEGORIES = 10
HIDDEN = 64


class DeepAE(nn.Module):
    """Architecture shared by the AE and DDAE checkpoints."""

    def __init__(self, latent_dim: int = 2, hidden: int = HIDDEN):
        super().__init__()
        encoder: list[nn.Module] = []
        in_dim = N_CATEGORIES
        for _ in range(4):
            encoder.extend(
                [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
            )
            in_dim = hidden
        encoder.append(nn.Linear(hidden, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(4):
            decoder.extend(
                [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
            )
            in_dim = hidden
        decoder.append(nn.Linear(hidden, N_CATEGORIES))
        self.decoder = nn.Sequential(*decoder)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_mean(x)
        return z, self.decoder(z)


class DeepVAE(nn.Module):
    """VAE using its posterior mean for stable dashboard coordinates."""

    def __init__(self, latent_dim: int = 2, hidden: int = HIDDEN):
        super().__init__()
        encoder: list[nn.Module] = []
        in_dim = N_CATEGORIES
        for _ in range(4):
            encoder.extend(
                [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
            )
            in_dim = hidden
        self.encoder = nn.Sequential(*encoder)
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)

        decoder: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(4):
            decoder.extend(
                [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
            )
            in_dim = hidden
        decoder.append(nn.Linear(hidden, N_CATEGORIES))
        self.decoder = nn.Sequential(*decoder)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_mu(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_mean(x)
        return z, self.decoder(z)


class GroupedMoE(nn.Module):
    """Inference-compatible copy of ``model/v2_ddae_moe/moe.py``."""

    def __init__(
        self,
        n_groups: int,
        in_dim: int,
        out_dim: int,
        n_experts: int = 4,
        top_k: int = 4,
        hidden_mult: int = 4,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        hidden = max(hidden_mult * out_dim, 4)

        self.w1 = nn.Parameter(
            torch.empty(n_groups, n_experts, in_dim, hidden)
        )
        self.b1 = nn.Parameter(torch.zeros(n_groups, n_experts, hidden))
        self.w2 = nn.Parameter(
            torch.empty(n_groups, n_experts, hidden, out_dim)
        )
        self.b2 = nn.Parameter(torch.zeros(n_groups, n_experts, out_dim))
        self.rw = nn.Parameter(torch.zeros(n_groups, in_dim, n_experts))
        self.rb = nn.Parameter(torch.zeros(n_groups, n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        grouped = x.view(batch, self.n_groups, self.in_dim)
        logits = torch.einsum("bgi,gie->bge", grouped, self.rw) + self.rb
        probs = logits.softmax(dim=-1)
        top_values, top_indices = probs.topk(self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        gate = torch.zeros_like(probs).scatter(-1, top_indices, top_values)

        hidden = torch.einsum("bgi,geih->bgeh", grouped, self.w1) + self.b1
        hidden = F.gelu(hidden)
        out = torch.einsum("bgeh,geho->bgeo", hidden, self.w2) + self.b2
        return (out * gate.unsqueeze(-1)).sum(dim=2).reshape(batch, -1)


class MoEEncoder(nn.Module):
    def __init__(self, latent_dim: int = 2, n_layers: int = 4):
        super().__init__()
        hidden = 40
        n_groups = 10
        group_dim = hidden // n_groups
        dims = [1] + [group_dim] * n_layers
        self.moes = nn.ModuleList(
            [
                GroupedMoE(n_groups, dims[i], group_dim)
                for i in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(n_layers)]
        )
        self.head = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for moe, norm in zip(self.moes, self.norms):
            hidden = F.gelu(norm(moe(hidden)))
        return self.head(hidden)


class MoEAE(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        hidden = 40
        self.encoder = MoEEncoder(latent_dim)
        decoder: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(4):
            decoder.extend(
                [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
            )
            in_dim = hidden
        decoder.append(nn.Linear(hidden, N_CATEGORIES))
        self.decoder = nn.Sequential(*decoder)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_mean(x)
        return z, self.decoder(z)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    short_label: str
    description: str
    checkpoint: Path
    architecture: str = "ae"


class ModelRegistry:
    """Lazily load checkpoints and cache their full-dataset embeddings."""

    def __init__(self, root: Path, counts: np.ndarray):
        self.root = root
        self.counts = np.asarray(counts, dtype=np.float32)
        self.specs = {
            spec.key: spec
            for spec in [
                ModelSpec(
                    "ae",
                    "AE",
                    "AE",
                    "只用 Poisson 重建損失的二維 Autoencoder。",
                    root / "hello/model_pt/ae_300.pt",
                ),
                ModelSpec(
                    "ae_fsce",
                    "AE + FSCE",
                    "AE + FSCE",
                    "重建之外，同時保持原始 POI 近鄰結構。",
                    root / "model/v2_deep_ae/result/ae_fsce.pt",
                ),
                ModelSpec(
                    "ddae",
                    "DAE",
                    "DAE",
                    "以破壞過的 count 輸入重建乾淨資料。",
                    root / "hello/model_pt/ddae_300.pt",
                ),
                ModelSpec(
                    "ddae_fsce",
                    "DAE + FSCE（主要方法）",
                    "DAE + FSCE",
                    "去噪重建與二維鄰近結構保持同時訓練。",
                    root / "model/v2_ddae_base/result/ae.pt",
                ),
                ModelSpec(
                    "vae",
                    "VAE",
                    "VAE",
                    "以 posterior mean 作為穩定的二維顯示座標。",
                    root / "model/v2_deep_vae/result/vae.pt",
                    architecture="vae",
                ),
                ModelSpec(
                    "ddae_moe",
                    "DAE + FSCE + MoE",
                    "MoE",
                    "以分組 Mixture-of-Experts encoder 建立二維座標。",
                    root / "model/v2_ddae_moe/result/ae.pt",
                    architecture="moe",
                ),
            ]
        }
        self._models: dict[str, nn.Module] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._stats: dict[str, dict[str, np.ndarray]] = {}
        self._lock = RLock()

    def public_specs(self) -> list[dict[str, object]]:
        return [
            {
                "key": spec.key,
                "label": spec.label,
                "short_label": spec.short_label,
                "description": spec.description,
                "available": spec.checkpoint.exists(),
            }
            for spec in self.specs.values()
        ]

    def _new_model(self, architecture: str) -> nn.Module:
        if architecture == "vae":
            return DeepVAE()
        if architecture == "moe":
            return MoEAE()
        return DeepAE()

    def get_model(self, key: str) -> nn.Module:
        if key not in self.specs:
            raise KeyError(f"未知模型：{key}")
        if key in self._models:
            return self._models[key]

        with self._lock:
            if key in self._models:
                return self._models[key]
            spec = self.specs[key]
            if not spec.checkpoint.exists():
                raise FileNotFoundError(f"找不到 checkpoint：{spec.checkpoint}")
            model = self._new_model(spec.architecture)
            state = torch.load(
                spec.checkpoint, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state, strict=True)
            model.eval()
            self._models[key] = model
            return model

    def encode(self, key: str, counts: np.ndarray) -> np.ndarray:
        model = self.get_model(key)
        array = np.asarray(counts, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(array), 256):
                tensor = torch.from_numpy(array[start : start + 256])
                chunks.append(model.encode_mean(tensor).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def embedding(self, key: str) -> np.ndarray:
        if key not in self._embeddings:
            with self._lock:
                if key not in self._embeddings:
                    model = self.get_model(key)
                    chunks: list[np.ndarray] = []
                    with torch.inference_mode():
                        for start in range(0, len(self.counts), 256):
                            tensor = torch.from_numpy(
                                self.counts[start : start + 256]
                            )
                            chunks.append(model.encode_mean(tensor).cpu().numpy())
                    embedding = np.concatenate(chunks, axis=0).astype(np.float32)
                    self._embeddings[key] = embedding
                    median = np.median(embedding, axis=0)
                    mad = np.median(np.abs(embedding - median), axis=0) * 1.4826
                    std = embedding.std(axis=0)
                    scale = np.where(
                        mad > 1e-6, mad, np.where(std > 1e-6, std, 1.0)
                    )
                    robust_dist = np.linalg.norm(
                        (embedding - median) / scale, axis=1
                    )
                    self._stats[key] = {
                        "median": median,
                        "scale": scale,
                        "sorted_dist": np.sort(robust_dist),
                    }
        return self._embeddings[key]

    def stats(self, key: str) -> dict[str, np.ndarray]:
        self.embedding(key)
        return self._stats[key]
