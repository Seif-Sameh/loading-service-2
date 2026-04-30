"""Inference-only adapter — wraps a trained PCT model as a :class:`PackingAlgorithm`.

Lazy-imports torch so the slim API container isn't forced to install it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.algorithms.base import PackingAlgorithm
from app.env.packing_env import PackingState


class PCTPackingAgent(PackingAlgorithm):
    """Greedy / sampled inference using a trained PCT (DRL_GAT) model."""

    code = "pct"
    display_name = "PCT (GAT + pointer)"

    def __init__(
        self,
        weights_path: str | os.PathLike,
        *,
        sample_actions: bool = False,
        device: str = "cpu",
    ) -> None:
        import torch

        from .pct_env import PCTEnvConfig
        from .pct_model import DRL_GAT, PCTConfig

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"PCT checkpoint missing: {weights_path}")
        ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)

        cfg = PCTConfig(**ckpt["pct_config"])
        env_cfg = PCTEnvConfig(**ckpt["env_config"]) if "env_config" in ckpt else PCTEnvConfig(
            internal_node_holder=cfg.internal_node_holder,
            leaf_node_holder=cfg.leaf_node_holder,
            internal_node_length=cfg.internal_node_length,
        )

        self._torch = torch
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.device = torch.device(device)
        self.sample_actions = sample_actions

        self.model = DRL_GAT(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    # ----- PackingAlgorithm contract -----

    def select(self, state: PackingState) -> int:
        if not state.candidates:
            return 0

        torch = self._torch
        cfg = self.cfg
        env_cfg = self.env_cfg
        graph_size = cfg.internal_node_holder + cfg.leaf_node_holder + 1
        max_dim = max(cfg.internal_node_length, 8, 6) + 2
        L = state.container.internal.length_mm
        W = state.container.internal.width_mm
        H = state.container.internal.height_mm

        obs = np.zeros((1, graph_size, max_dim), dtype=np.float32)

        # internal nodes
        n_placed = min(len(state.placements), cfg.internal_node_holder)
        for i, p in enumerate(state.placements[:n_placed]):
            obs[0, i, 0] = p.position.x_mm / L
            obs[0, i, 1] = p.position.y_mm / H
            obs[0, i, 2] = p.position.z_mm / W
            obs[0, i, 3] = p.rotated_dimensions.length_mm / L
            obs[0, i, 4] = p.rotated_dimensions.height_mm / H
            obs[0, i, 5] = p.rotated_dimensions.width_mm / W

        # leaf nodes
        from app.schemas import Rotation

        n_leaves = min(len(state.candidates), cfg.leaf_node_holder)
        for i, c in enumerate(state.candidates[:n_leaves]):
            row = cfg.internal_node_holder + i
            obs[0, row, 0] = c.position.x_mm / L
            obs[0, row, 1] = c.position.y_mm / H
            obs[0, row, 2] = c.position.z_mm / W
            obs[0, row, 3] = c.rotated_dimensions.length_mm / L
            obs[0, row, 4] = c.rotated_dimensions.height_mm / H
            obs[0, row, 5] = c.rotated_dimensions.width_mm / W
            obs[0, row, 6] = 1.0 if c.rotation == Rotation.LWH else 0.0
            obs[0, row, 7] = 1.0 if c.rotation == Rotation.WLH else 0.0
            obs[0, row, 8] = 1.0

        # next item
        next_row = cfg.internal_node_holder + cfg.leaf_node_holder
        if state.current_item is not None:
            d = state.current_item.dimensions
            obs[0, next_row, 0] = d.length_mm / L
            obs[0, next_row, 1] = d.height_mm / H
            obs[0, next_row, 2] = d.width_mm / W

        # full mask
        obs[0, :n_placed, -1] = 1.0
        obs[0, cfg.internal_node_holder : cfg.internal_node_holder + n_leaves, -1] = 1.0
        if state.current_item is not None:
            obs[0, next_row, -1] = 1.0

        with torch.no_grad():
            t_obs = torch.from_numpy(obs).to(self.device)
            _, action, _, _ = self.model(t_obs, deterministic=not self.sample_actions, evaluate=True)
            cand_idx = int(action.item())

        if cand_idx >= len(state.candidates):
            cand_idx = 0
        return cand_idx
