"""Preference-conditioned PCT model.

Extends the base PCT actor with a preference-vector input that the policy uses to
condition its placement decisions on operator priorities. At training time the
preference is sampled randomly from a Dirichlet distribution each episode; at
inference time the operator picks it via the UI.

The architectural change is small: we add a preference-embedding MLP that produces a
``embedding_dim``-sized vector, which is broadcast across all graph nodes and added to
each node's embedding *after* the input projection MLPs and *before* the GAT layers.
This is the standard preference-conditioning pattern from MORL literature (Yang et al.
2019, Abels et al. 2019, Basaklar et al. 2023, PD-MORL).

The critic head is replaced with a *multi-head value function* — one scalar per
objective. PPO uses the scalarised value `w · V` exactly like the scalarised reward.
This is the value-decomposed multi-objective formulation from Yang et al. 2019.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from .attention_model import AttentionModel
from .pct_model import PCTConfig
from .utils import init

N_OBJECTIVES = 5


@dataclass
class PCTMORLConfig(PCTConfig):
    """Inherits PCTConfig and adds the number of objectives the policy conditions on."""

    n_objectives: int = N_OBJECTIVES


class _PreferenceEmbed(nn.Module):
    """Maps a preference vector (B, n_objectives) → (B, embedding_dim)."""

    def __init__(self, n_objectives: int, embedding_dim: int) -> None:
        super().__init__()
        gain = nn.init.calculate_gain("leaky_relu")
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
        self.net = nn.Sequential(
            init_(nn.Linear(n_objectives, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, embedding_dim)),
        )

    def forward(self, pref: torch.Tensor) -> torch.Tensor:
        return self.net(pref)


class MORLAttentionModel(AttentionModel):
    """Drop-in MORL extension. Same actor mechanics, but each node embedding is shifted
    by the preference embedding before being fed to the GAT."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_encode_layers: int = 2,
        tanh_clipping: float = 10.0,
        mask_inner: bool = False,
        mask_logits: bool = False,
        n_heads: int = 1,
        internal_node_holder: int | None = None,
        internal_node_length: int | None = None,
        leaf_node_holder: int | None = None,
        n_objectives: int = N_OBJECTIVES,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_encode_layers=n_encode_layers,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            n_heads=n_heads,
            internal_node_holder=internal_node_holder,
            internal_node_length=internal_node_length,
            leaf_node_holder=leaf_node_holder,
        )
        self.n_objectives = n_objectives
        self.pref_embed = _PreferenceEmbed(n_objectives, embedding_dim)

    def forward(
        self,
        observation: torch.Tensor,
        preference: torch.Tensor,
        deterministic: bool = False,
        evaluate_action: bool = False,
        normFactor: float = 1.0,
        evaluate: bool = False,
    ):
        """Same return signature as AttentionModel.forward, but takes preference."""
        from .utils import observation_decode_leaf_node

        internal_nodes, leaf_nodes, next_item, leaf_valid, full_mask = observation_decode_leaf_node(
            observation,
            self.internal_node_holder,
            self.internal_node_length,
            self.leaf_node_holder,
        )
        leaf_node_mask = 1 - leaf_valid
        valid_length = full_mask.sum(1)
        full_mask_inv = 1 - full_mask

        batch_size = observation.size(0)
        graph_size = observation.size(1)
        internal_nodes_size = internal_nodes.size(1)
        leaf_node_size = leaf_nodes.size(1)
        next_size = next_item.size(1)

        internal_inputs = (
            internal_nodes.contiguous().view(batch_size * internal_nodes_size, self.internal_node_length)
            * normFactor
        )
        leaf_inputs = leaf_nodes.contiguous().view(batch_size * leaf_node_size, 8) * normFactor
        current_inputs = next_item.contiguous().view(batch_size * next_size, 6) * normFactor

        internal_emb = self.init_internal_node_embed(internal_inputs).reshape(
            batch_size, -1, self.embedding_dim
        )
        leaf_emb = self.init_leaf_node_embed(leaf_inputs).reshape(
            batch_size, -1, self.embedding_dim
        )
        next_emb = self.init_next_embed(current_inputs).reshape(batch_size, -1, self.embedding_dim)
        init_emb = torch.cat((internal_emb, leaf_emb, next_emb), dim=1)  # (B, graph_size, embed)

        # === MORL conditioning: add preference embedding to every node ===
        pref_features = self.pref_embed(preference)         # (B, embed)
        pref_features = pref_features.unsqueeze(1)          # (B, 1, embed)
        init_emb = init_emb + pref_features                  # broadcast across graph_size

        init_emb = init_emb.view(batch_size * graph_size, self.embedding_dim)

        embeddings, _ = self.embedder(init_emb, mask=full_mask_inv, evaluate=evaluate)
        embed_shape = (batch_size, graph_size, embeddings.shape[-1])

        log_p, action_log_prob, pointers, dist_entropy, dist, hidden = self._inner(
            embeddings,
            deterministic=deterministic,
            evaluate_action=evaluate_action,
            shape=embed_shape,
            mask=leaf_node_mask,
            full_mask=full_mask_inv,
            valid_length=valid_length,
        )
        return action_log_prob, pointers, dist_entropy, hidden, dist


class MORL_DRL_GAT(nn.Module):
    """Top-level MORL actor-critic. Critic outputs one scalar value per objective."""

    def __init__(self, cfg: PCTMORLConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.actor = MORLAttentionModel(
            embedding_dim=cfg.embedding_size,
            hidden_dim=cfg.hidden_size,
            n_encode_layers=cfg.gat_layer_num,
            n_heads=cfg.n_heads,
            internal_node_holder=cfg.internal_node_holder,
            internal_node_length=cfg.internal_node_length,
            leaf_node_holder=cfg.leaf_node_holder,
            n_objectives=cfg.n_objectives,
        )
        gain = sqrt(2.0)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
        # One critic head per objective so PPO can compute per-objective advantages and
        # then scalarise; this is the value-decomposed MORL formulation.
        self.critic = init_(nn.Linear(cfg.embedding_size, cfg.n_objectives))

    def forward(
        self,
        observation: torch.Tensor,
        preference: torch.Tensor,
        deterministic: bool = False,
        normFactor: float = 1.0,
        evaluate: bool = False,
    ):
        """Returns (action_log_prob, action, entropy, value_vec) where value_vec is (B, n_obj)."""
        out, action, dist_entropy, hidden, _ = self.actor(
            observation, preference,
            deterministic=deterministic, normFactor=normFactor, evaluate=evaluate,
        )
        value_vec = self.critic(hidden)
        return out, action, dist_entropy, value_vec

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        preference: torch.Tensor,
        actions: torch.Tensor,
        normFactor: float = 1.0,
    ):
        _, _, dist_entropy, hidden, dist = self.actor(
            observation, preference, evaluate_action=True, normFactor=normFactor,
        )
        action_log_probs = dist.log_probs(actions)
        value_vec = self.critic(hidden)
        return value_vec, action_log_probs, dist_entropy.mean()
