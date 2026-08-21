"""Data preparation and simulation logic for the POI latent dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model_runtime import ModelRegistry


CATEGORY_NAMES = [
    "餐飲",
    "零售",
    "夜生活",
    "社區／政府",
    "交通",
    "商業服務",
    "地標／戶外",
    "藝文娛樂",
    "醫療",
    "運動休閒",
]

CATEGORY_COLORS = [
    "#e05a47",
    "#2f9d78",
    "#8553a6",
    "#456dc0",
    "#ed913c",
    "#35a9b8",
    "#12756f",
    "#c64e92",
    "#94603d",
    "#7b7d35",
]


class DashboardData:
    def __init__(self, root: Path):
        self.root = root
        patch_path = root / "data/patch/patches.npz"
        if not patch_path.exists():
            raise FileNotFoundError(f"找不到 patch 資料：{patch_path}")

        with np.load(patch_path) as patches:
            self.dx = patches["dx"].astype(np.float32)
            self.dy = patches["dy"].astype(np.float32)
            self.cat = patches["cat"].astype(np.int64)
            self.offsets = patches["offsets"].astype(np.int64)
            self.lat = patches["center_lat"].astype(np.float64)
            self.lon = patches["center_lon"].astype(np.float64)
            self.n_poi = patches["n_poi"].astype(np.int64)

        self.counts = np.zeros((len(self.n_poi), len(CATEGORY_NAMES)), dtype=np.float32)
        patch_ids = np.repeat(np.arange(len(self.n_poi)), np.diff(self.offsets))
        np.add.at(self.counts, (patch_ids, self.cat), 1)
        self.registry = ModelRegistry(root, self.counts)
        self.presets = self._build_presets()

    @property
    def n_patches(self) -> int:
        return len(self.n_poi)

    def _nearest_patch(self, latitude: float, longitude: float) -> int:
        lon_scale = np.cos(np.deg2rad(latitude))
        distance = np.sqrt(
            (self.lat - latitude) ** 2
            + ((self.lon - longitude) * lon_scale) ** 2
        )
        return int(np.argmin(distance))

    def _build_presets(self) -> list[dict[str, object]]:
        named_centres = [
            ("東京站周邊", 35.6812, 139.7671),
            ("新宿站周邊", 35.6896, 139.7006),
            ("澀谷站周邊", 35.6580, 139.7016),
        ]
        presets: list[dict[str, object]] = []
        used: set[int] = set()
        for label, latitude, longitude in named_centres:
            patch_id = self._nearest_patch(latitude, longitude)
            if patch_id not in used:
                presets.append(
                    {
                        "id": f"centre-{patch_id}",
                        "label": label,
                        "group": "市中心樣本",
                        "patch_id": patch_id,
                    }
                )
                used.add(patch_id)

        low_threshold = float(np.percentile(self.n_poi, 25))
        candidates = np.flatnonzero(self.n_poi <= low_threshold)
        directions = [
            ("西側低密度樣本", self.lon[candidates], "min"),
            ("北側低密度樣本", self.lat[candidates], "max"),
            ("南側低密度樣本", self.lat[candidates], "min"),
        ]
        for label, values, operation in directions:
            order = np.argsort(values)
            if operation == "max":
                order = order[::-1]
            patch_id = next(
                int(candidates[i]) for i in order if int(candidates[i]) not in used
            )
            presets.append(
                {
                    "id": f"low-{patch_id}",
                    "label": label,
                    "group": "相對低密度樣本",
                    "patch_id": patch_id,
                }
            )
            used.add(patch_id)
        return presets

    def bootstrap(self) -> dict[str, object]:
        global_counts = self.counts.sum(axis=0).astype(int)
        return {
            "categories": [
                {"id": i, "name": name, "color": CATEGORY_COLORS[i]}
                for i, name in enumerate(CATEGORY_NAMES)
            ],
            "models": self.registry.public_specs(),
            "presets": self.presets,
            "dataset": {
                "patch_count": self.n_patches,
                "poi_records": int(self.n_poi.sum()),
                "median_poi": float(np.median(self.n_poi)),
                "min_poi": int(self.n_poi.min()),
                "max_poi": int(self.n_poi.max()),
                "global_counts": global_counts.tolist(),
                "scope_note": (
                    "資料只包含至少 10 個 POI 的 100m × 100m patch；"
                    "低密度樣本是資料範圍內的相對比較，不代表真正鄉村。"
                ),
            },
        }

    def patch(self, patch_id: int) -> dict[str, object]:
        if patch_id < 0 or patch_id >= self.n_patches:
            raise ValueError(f"patch_id 必須介於 0 和 {self.n_patches - 1}")
        start, end = self.offsets[patch_id : patch_id + 2]
        counts = self.counts[patch_id].astype(int)
        total = int(counts.sum())
        density_percentile = float(
            100 * np.mean(self.n_poi <= self.n_poi[patch_id])
        )
        top_category = int(np.argmax(counts))
        return {
            "id": patch_id,
            "latitude": round(float(self.lat[patch_id]), 6),
            "longitude": round(float(self.lon[patch_id]), 6),
            "total": total,
            "density_percentile": round(density_percentile, 1),
            "density_label": self._density_label(density_percentile),
            "top_category": CATEGORY_NAMES[top_category],
            "counts": counts.tolist(),
            "ratios": (counts / max(total, 1)).round(4).tolist(),
            "points": [
                {
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "category": int(category),
                }
                for x, y, category in zip(
                    self.dx[start:end], self.dy[start:end], self.cat[start:end]
                )
            ],
        }

    @staticmethod
    def _density_label(percentile: float) -> str:
        if percentile >= 80:
            return "高密度"
        if percentile <= 30:
            return "相對低密度"
        return "中密度"

    def embedding(self, model_key: str) -> dict[str, object]:
        embedding = self.registry.embedding(model_key)
        spec = self.registry.specs[model_key]
        return {
            "model": model_key,
            "label": spec.label,
            "description": spec.description,
            "points": np.round(embedding, 5).tolist(),
        }

    def simulate(
        self, model_key: str, patch_id: int, category: int, amount: int
    ) -> dict[str, object]:
        if model_key not in self.registry.specs:
            raise ValueError(f"未知模型：{model_key}")
        if patch_id < 0 or patch_id >= self.n_patches:
            raise ValueError(f"patch_id 必須介於 0 和 {self.n_patches - 1}")
        if category < 0 or category >= len(CATEGORY_NAMES):
            raise ValueError("category 必須介於 0 和 9")
        if amount < 0 or amount > 200:
            raise ValueError("加入數量必須介於 0 和 200")

        if amount <= 50:
            amounts = np.arange(amount + 1, dtype=np.int64)
        else:
            amounts = np.unique(
                np.rint(np.linspace(0, amount, 51)).astype(np.int64)
            )
        curve_counts = np.repeat(self.counts[patch_id][None, :], len(amounts), axis=0)
        curve_counts[:, category] += amounts
        path = self.registry.encode(model_key, curve_counts)
        stats = self.registry.stats(model_key)
        scale = stats["scale"]
        median = stats["median"]
        sorted_dist = stats["sorted_dist"]

        shift = np.linalg.norm((path - path[0]) / scale, axis=1)
        position_dist = np.linalg.norm((path - median) / scale, axis=1)
        final_percentile = float(
            100
            * np.searchsorted(sorted_dist, position_dist[-1], side="right")
            / len(sorted_dist)
        )

        if len(amounts) > 1:
            step = np.diff(amounts).astype(np.float64)
            marginal = np.linalg.norm(np.diff(path / scale, axis=0), axis=1) / step
            initial_response = float(marginal[0])
            final_response = float(marginal[-1])
            if initial_response > 1e-9:
                saturation = float(
                    np.clip(1 - final_response / initial_response, 0, 1) * 100
                )
            else:
                saturation = 0.0
        else:
            initial_response = final_response = saturation = 0.0

        updated_counts = self.counts[patch_id].copy()
        updated_counts[category] += amount
        return {
            "model": model_key,
            "patch_id": patch_id,
            "category": category,
            "category_name": CATEGORY_NAMES[category],
            "amount": amount,
            "base_counts": self.counts[patch_id].astype(int).tolist(),
            "updated_counts": updated_counts.astype(int).tolist(),
            "path": [
                {
                    "amount": int(a),
                    "x": round(float(z[0]), 6),
                    "y": round(float(z[1]), 6),
                    "shift": round(float(s), 6),
                }
                for a, z, s in zip(amounts, path, shift)
            ],
            "metrics": {
                "standardized_shift": round(float(shift[-1]), 4),
                "latent_percentile": round(final_percentile, 1),
                "initial_response": round(initial_response, 5),
                "final_response": round(final_response, 5),
                "saturation_index": round(saturation, 1),
            },
            "interpretation": self._interpret(final_percentile, saturation),
        }

    @staticmethod
    def _interpret(percentile: float, saturation: float) -> str:
        rarity = (
            "已進入非常少見的 latent 區域"
            if percentile >= 95
            else "位於偏少見的 latent 區域"
            if percentile >= 80
            else "仍位於常見的 latent 範圍"
        )
        response = (
            "且模型反應明顯趨緩"
            if saturation >= 60
            else "且模型反應略有趨緩"
            if saturation >= 25
            else "且尚未出現明顯反應遞減"
        )
        return f"{rarity}，{response}。"

