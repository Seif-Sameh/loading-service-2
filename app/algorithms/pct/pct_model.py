"""DRL-GAT actor-critic — top-level PCT policy.

Ported from https://github.com/alexfrom0815/Online-3D-BPP-PCT/blob/master/model.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from .attention_model import AttentionModel
from .utils import init


@dataclass
class PCTConfig:
    """Hyperparameters for the PCT actor-critic.

    Defaults match the paper's ``Setting 1`` (with-stability discrete bins): 80 internal
    node holders, 50 leaf node holders, 64-D embeddings, 1 GAT layer.
    """

    embedding_size: int = 64
    hidden_size: int = 128
    gat_layer_num: int = 1
    n_heads: int = 1
    internal_node_holder: int = 80
    leaf_node_holder: int = 50
    # Each placed (internal) node is described by 6 numbers by default
    # (position xyz + size xyz). Increase if you append density / fragility / ID etc.
    internal_node_length: int = 6


class DRL_GAT(nn.Module):
    """Top-level actor-critic. Returns ``(log_prob, action, entropy, value)``."""

    def __init__(self, cfg: PCTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.actor = AttentionModel(
            embedding_dim=cfg.embedding_size,
            hidden_dim=cfg.hidden_size,
            n_encode_layers=cfg.gat_layer_num,
            n_heads=cfg.n_heads,
            internal_node_holder=cfg.internal_node_holder,
            internal_node_length=cfg.internal_node_length,
            leaf_node_holder=cfg.leaf_node_holder,
        )

        gain = sqrt(2.0)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
        self.critic = init_(nn.Linear(cfg.embedding_size, 1))

    def forward(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        normFactor: float = 1.0,
        evaluate: bool = False,
    ):
        out, action, dist_entropy, hidden, _ = self.actor(
            observation, deterministic=deterministic, normFactor=normFactor, evaluate=evaluate
        )
        value = self.critic(hidden)
        return out, action, dist_entropy, value

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
        normFactor: float = 1.0,
    ):
        """Re-evaluate stored actions during a PPO/ACKTR update."""
        _, _, dist_entropy, hidden, dist = self.actor(
            observation, evaluate_action=True, normFactor=normFactor
        )
        action_log_probs = dist.log_probs(actions)
        value = self.critic(hidden)
        return value, action_log_probs, dist_entropy.mean()


# ---------------------------------------------------------------------------
# Helper: build the packed observation tensor used by the model
# ---------------------------------------------------------------------------


def pack_observation(
    internal_nodes: torch.Tensor,   # (B, internal_node_holder, internal_node_length)
    leaf_nodes: torch.Tensor,       # (B, leaf_node_holder, 8)
    next_item: torch.Tensor,        # (B, 1, 6)
    leaf_valid: torch.Tensor,       # (B, leaf_node_holder)
    real_mask: torch.Tensor,        # (B, internal_node_holder + leaf_node_holder + 1)
    *,
    cfg: PCTConfig,
) -> torch.Tensor:
    """Pack the four sub-tensors into the single (B, graph_size, max_dim) layout
    that :func:`observation_decode_leaf_node` expects."""
    B = internal_nodes.size(0)
    graph_size = cfg.internal_node_holder + cfg.leaf_node_holder + 1
    # max feature width: max(internal_node_length, 8, 6, 1+1 trailing flags)
    max_dim = max(cfg.internal_node_length, 8, 6) + 2  # leaf-validity col @ 8, full_mask col @ -1
    out = internal_nodes.new_zeros((B, graph_size, max_dim))
    # internal nodes [0 : H_i, 0 : L_i]
    out[:, : cfg.internal_node_holder, : cfg.internal_node_length] = internal_nodes
    # leaf nodes [H_i : H_i+H_l, 0 : 8]
    out[:, cfg.internal_node_holder : cfg.internal_node_holder + cfg.leaf_node_holder, :8] = leaf_nodes
    # leaf validity flag at column 8
    out[
        :,
        cfg.internal_node_holder : cfg.internal_node_holder + cfg.leaf_node_holder,
        8,
    ] = leaf_valid
    # next item [last 1, 0 : 6]
    out[:, -1:, :6] = next_item
    # real-node mask in last column for every row
    out[:, :, -1] = real_mask
    return out
