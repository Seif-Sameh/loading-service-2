"""Inference-time MORL PCT agent — operator dials a preference, model returns matched policy."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.algorithms.base import PackingAlgorithm
from app.env.packing_env import PackingState


class MORLPCTAgent(PackingAlgorithm):
    """Wraps a trained MORL_DRL_GAT as a PackingAlgorithm.

    Pass ``preference`` at construction (a 5-element list summing to 1) to dial the
    operator's trade-off. To sweep the Pareto front at evaluation time, create one
    agent per preference vector and run them across the same voyages.
    """

    code = "pct_morl"
    display_name = "MORL-PCT (preference-conditioned)"

    def __init__(
        self,
        weights_path: str | os.PathLike,
        *,
        preference: list[float] | np.ndarray = (0.5, 0.125, 0.125, 0.125, 0.125),
        deterministic: bool = True,
        device: str = "cpu",
    ) -> None:
        import torch

        from .morl_model import MORL_DRL_GAT, PCTMORLConfig
        from .pct_env import PCTEnvConfig

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"MORL checkpoint missing: {weights_path}")
        ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)

        cfg = PCTMORLConfig(**ckpt["pct_config"])
        env_cfg = (
            PCTEnvConfig(**ckpt["env_config"])
            if "env_config" in ckpt
            else PCTEnvConfig(
                internal_node_holder=cfg.internal_node_holder,
                leaf_node_holder=cfg.leaf_node_holder,
                internal_node_length=cfg.internal_node_length,
            )
        )

        self._torch = torch
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.device = torch.device(device)
        self.deterministic = deterministic

        # Validate preference: non-negative, sums to 1.
        pref = np.asarray(preference, dtype=np.float32)
        if pref.shape != (cfg.n_objectives,):
            raise ValueError(f"preference must have shape ({cfg.n_objectives},), got {pref.shape}")
        if (pref < 0).any():
            raise ValueError("preference entries must be non-negative")
        s = pref.sum()
        if s <= 0:
            raise ValueError("preference cannot be the zero vector")
        self.preference = pref / s

        self.model = MORL_DRL_GAT(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    # ----- PackingAlgorithm contract -----

    def select(self, state: PackingState) -> int:
        if not state.candidates:
            return 0

        torch = self._torch
        cfg = self.cfg
        graph_size = cfg.internal_node_holder + cfg.leaf_node_holder + 1
        max_dim = max(cfg.internal_node_length, 8, 6) + 2
        L = state.container.internal.length_mm
        W = state.container.internal.width_mm
        H = state.container.internal.height_mm

        obs = np.zeros((1, graph_size, max_dim), dtype=np.float32)
        n_placed = min(len(state.placements), cfg.internal_node_holder)
        for i, p in enumerate(state.placements[:n_placed]):
            obs[0, i, 0] = p.position.x_mm / L
            obs[0, i, 1] = p.position.y_mm / H
            obs[0, i, 2] = p.position.z_mm / W
            obs[0, i, 3] = p.rotated_dimensions.length_mm / L
            obs[0, i, 4] = p.rotated_dimensions.height_mm / H
            obs[0, i, 5] = p.rotated_dimensions.width_mm / W

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

        next_row = cfg.internal_node_holder + cfg.leaf_node_holder
        if state.current_item is not None:
            d = state.current_item.dimensions
            obs[0, next_row, 0] = d.length_mm / L
            obs[0, next_row, 1] = d.height_mm / H
            obs[0, next_row, 2] = d.width_mm / W

        obs[0, :n_placed, -1] = 1.0
        obs[0, cfg.internal_node_holder : cfg.internal_node_holder + n_leaves, -1] = 1.0
        if state.current_item is not None:
            obs[0, next_row, -1] = 1.0

        pref = np.asarray(self.preference, dtype=np.float32)[None, :]

        with torch.no_grad():
            t_obs = torch.from_numpy(obs).to(self.device)
            t_pref = torch.from_numpy(pref).to(self.device)
            _, action, _, _ = self.model(t_obs, t_pref, deterministic=self.deterministic, evaluate=True)
            cand_idx = int(action.item())

        if cand_idx >= len(state.candidates):
            cand_idx = 0
        return cand_idx
