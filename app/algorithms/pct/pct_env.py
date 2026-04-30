"""Gymnasium environment that produces PCT-shaped observations.

Wraps our existing :class:`PackingEnv` (heightmap + constraint layer + corner-point
candidate generator) and translates its state into the packed (graph_size, max_dim)
observation tensor PCT's :class:`AttentionModel` expects.

Why wrap rather than re-implement: our constraint layer (weight, IMDG, reefer,
floor-load, orientation lock) is non-trivial and already tested. The PCT paper's
own env handles only basic non-overlap + stability; we get those plus full
operational constraints for free.

Action: the model outputs a leaf-node index in ``[0, leaf_node_holder)``. We
map it directly to ``state.candidates[action]`` — the env's candidate list is
the source of truth for "this leaf is feasible". Invalid leaves (beyond the
real candidate count) are masked out before the model samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.env.packing_env import PackingEnv
from app.schemas import CargoItem, Container, Rotation
from .pct_model import PCTConfig


@dataclass
class PCTEnvConfig:
    """Knobs for :class:`PCTEnv` — names match the upstream paper."""

    internal_node_holder: int = 80
    leaf_node_holder: int = 50
    internal_node_length: int = 6  # x, y, z, l, h, w (normalised)
    heightmap_resolution_mm: int = 10
    # max_candidates fed to the underlying PackingEnv; should be >= leaf_node_holder
    # so we can fill the leaf bucket even when many candidates exist.
    max_candidates: int = 50

    def to_pct_config(self) -> PCTConfig:
        return PCTConfig(
            embedding_size=64,
            hidden_size=128,
            gat_layer_num=1,
            n_heads=1,
            internal_node_holder=self.internal_node_holder,
            leaf_node_holder=self.leaf_node_holder,
            internal_node_length=self.internal_node_length,
        )


class PCTEnv:
    """Thin adapter that exposes a PCT-flavored observation around our PackingEnv.

    Not a full ``gymnasium.Env`` subclass on purpose — we don't need the gym registry,
    just the step/reset interface our PPO trainer consumes.
    """

    def __init__(
        self,
        container: Container,
        items: list[CargoItem],
        cfg: PCTEnvConfig | None = None,
    ) -> None:
        self.cfg = cfg or PCTEnvConfig()
        self._inner = PackingEnv(
            container=container,
            items=items,
            heightmap_resolution_mm=self.cfg.heightmap_resolution_mm,
            max_candidates=max(self.cfg.max_candidates, self.cfg.leaf_node_holder),
        )

    # ---------- public ----------

    @property
    def container(self) -> Container:
        return self._inner.container

    @property
    def state(self):
        return self._inner.state

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._inner.reset()
        return self._build_observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        n_real = len(self._inner.state.candidates)
        # Action mask should have prevented an out-of-range choice; clamp defensively.
        cand_idx = action if 0 <= action < n_real else 0
        _, reward, done, truncated, info = self._inner.step(cand_idx)
        return self._build_observation(), float(reward), bool(done), bool(truncated), self._info() | info

    def final_score(self):
        return self._inner.final_score()

    @property
    def graph_size(self) -> int:
        return self.cfg.internal_node_holder + self.cfg.leaf_node_holder + 1

    @property
    def max_feature_dim(self) -> int:
        # Internal length, leaf 8 cols, next-item 6 cols + leaf_valid col + full_mask col.
        return max(self.cfg.internal_node_length, 8, 6) + 2

    # ---------- internals ----------

    def _info(self) -> dict[str, Any]:
        st = self._inner.state
        return {
            "n_placed": len(st.placements),
            "n_remaining": len(st.items_remaining),
            "n_candidates": len(st.candidates),
        }

    def _build_observation(self) -> np.ndarray:
        cfg = self.cfg
        st = self._inner.state
        cont = st.container
        L = cont.internal.length_mm
        W = cont.internal.width_mm
        H = cont.internal.height_mm

        graph_size = self.graph_size
        max_dim = self.max_feature_dim
        out = np.zeros((graph_size, max_dim), dtype=np.float32)

        # ----- internal nodes (placed items) -----
        items_by_id = {it.id: it for it in self._inner.items}
        n_placed = min(len(st.placements), cfg.internal_node_holder)
        for i, p in enumerate(st.placements[:n_placed]):
            out[i, 0] = p.position.x_mm / L
            out[i, 1] = p.position.y_mm / H
            out[i, 2] = p.position.z_mm / W
            out[i, 3] = p.rotated_dimensions.length_mm / L
            out[i, 4] = p.rotated_dimensions.height_mm / H
            out[i, 5] = p.rotated_dimensions.width_mm / W

        # ----- leaf nodes (candidates) -----
        n_leaves = min(len(st.candidates), cfg.leaf_node_holder)
        for i, c in enumerate(st.candidates[:n_leaves]):
            row = cfg.internal_node_holder + i
            out[row, 0] = c.position.x_mm / L
            out[row, 1] = c.position.y_mm / H
            out[row, 2] = c.position.z_mm / W
            out[row, 3] = c.rotated_dimensions.length_mm / L
            out[row, 4] = c.rotated_dimensions.height_mm / H
            out[row, 5] = c.rotated_dimensions.width_mm / W
            # Two-bit one-hot orientation indicator (LWH vs WLH).
            out[row, 6] = 1.0 if c.rotation == Rotation.LWH else 0.0
            out[row, 7] = 1.0 if c.rotation == Rotation.WLH else 0.0
            out[row, 8] = 1.0  # leaf_valid flag

        # ----- next item -----
        next_row = cfg.internal_node_holder + cfg.leaf_node_holder
        if st.current_item is not None:
            d = st.current_item.dimensions
            out[next_row, 0] = d.length_mm / L
            out[next_row, 1] = d.height_mm / H
            out[next_row, 2] = d.width_mm / W

        # ----- full mask (real-node bitmap, last column) -----
        out[:n_placed, -1] = 1.0
        out[
            cfg.internal_node_holder : cfg.internal_node_holder + n_leaves, -1
        ] = 1.0
        if st.current_item is not None:
            out[next_row, -1] = 1.0

        return out
