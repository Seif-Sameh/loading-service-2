"""PCT actor — GAT encoder + pointer attention head.

Ported with light cleanups from
https://github.com/alexfrom0815/Online-3D-BPP-PCT/blob/master/attention_model.py.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from .distributions import FixedCategorical
from .graph_encoder import GraphAttentionEncoder
from .utils import init, observation_decode_leaf_node


class _AttentionModelFixed(NamedTuple):
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor


class AttentionModel(nn.Module):
    """Actor: encodes the PCT graph and produces a categorical over leaf nodes."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,                 # kept for compatibility; not used directly
        n_encode_layers: int = 2,
        tanh_clipping: float = 10.0,
        mask_inner: bool = False,
        mask_logits: bool = False,
        n_heads: int = 1,
        internal_node_holder: int | None = None,
        internal_node_length: int | None = None,
        leaf_node_holder: int | None = None,
    ) -> None:
        super().__init__()
        assert internal_node_holder is not None
        assert leaf_node_holder is not None
        assert internal_node_length is not None

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.n_heads = n_heads

        self.internal_node_holder = internal_node_holder
        self.internal_node_length = internal_node_length
        self.leaf_node_holder = leaf_node_holder
        self.next_holder = 1

        graph_size = internal_node_holder + leaf_node_holder + self.next_holder
        gain = nn.init.calculate_gain("leaky_relu")
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        # Three independent MLPs project heterogeneous descriptors to a shared embedding.
        self.init_internal_node_embed = nn.Sequential(
            init_(nn.Linear(internal_node_length, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, embedding_dim)),
        )
        self.init_leaf_node_embed = nn.Sequential(
            init_(nn.Linear(8, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, embedding_dim)),
        )
        self.init_next_embed = nn.Sequential(
            init_(nn.Linear(6, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, embedding_dim)),
        )

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
            graph_size=graph_size,
        )

        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0

    # ------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------

    def forward(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        evaluate_action: bool = False,
        normFactor: float = 1.0,
        evaluate: bool = False,
    ):
        """Return ``(action_log_probs, action, dist_entropy, hidden, dist)``."""
        internal_nodes, leaf_nodes, next_item, leaf_valid, full_mask = observation_decode_leaf_node(
            observation,
            self.internal_node_holder,
            self.internal_node_length,
            self.leaf_node_holder,
        )
        leaf_node_mask = 1 - leaf_valid       # 1 = invalid (to be masked out)
        valid_length = full_mask.sum(1)
        full_mask_inv = 1 - full_mask         # 1 = padding row (to be masked out)

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

        # Project each subset of nodes to the shared embedding space.
        internal_emb = self.init_internal_node_embed(internal_inputs).reshape(
            batch_size, -1, self.embedding_dim
        )
        leaf_emb = self.init_leaf_node_embed(leaf_inputs).reshape(
            batch_size, -1, self.embedding_dim
        )
        next_emb = self.init_next_embed(current_inputs).reshape(batch_size, -1, self.embedding_dim)
        init_emb = torch.cat((internal_emb, leaf_emb, next_emb), dim=1).view(
            batch_size * graph_size, self.embedding_dim
        )

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

    # ------------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------------

    def _inner(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool,
        evaluate_action: bool,
        shape,
        full_mask: torch.Tensor,
        valid_length: torch.Tensor,
    ):
        fixed = self._precompute(embeddings, shape=shape, full_mask=full_mask, valid_length=valid_length)
        log_p, mask = self._get_log_p(fixed, mask)

        if deterministic:
            masked_outs = log_p * (1 - mask)
            if torch.sum(masked_outs) == 0:
                masked_outs = masked_outs + 1e-20
        else:
            masked_outs = log_p * (1 - mask) + 1e-20
        log_p = torch.div(masked_outs, torch.sum(masked_outs, dim=1).unsqueeze(1))

        dist = FixedCategorical(probs=log_p)
        dist_entropy = dist.entropy()

        if deterministic:
            selected = dist.mode()
        else:
            selected = dist.sample()

        action_log_probs = None if evaluate_action else dist.log_probs(selected)
        return log_p, action_log_probs, selected, dist_entropy, dist, fixed.context_node_projected

    def _precompute(self, embeddings: torch.Tensor, *, shape, full_mask: torch.Tensor, valid_length: torch.Tensor):
        trans = embeddings.view(shape)
        full_mask_3d = full_mask.view(shape[0], shape[1], 1).expand(shape).bool()
        trans = trans.masked_fill(full_mask_3d, 0.0)
        graph_embed = trans.sum(1)
        trans = trans.view(embeddings.shape)

        graph_embed = graph_embed / valid_length.reshape(-1, 1)
        fixed_context = self.project_fixed_context(graph_embed)

        glimpse_k_fixed, glimpse_v_fixed, logit_k_fixed = (
            self.project_node_embeddings(trans).view((shape[0], 1, shape[1], -1)).chunk(3, dim=-1)
        )
        return _AttentionModelFixed(
            trans,
            fixed_context,
            self._make_heads(glimpse_k_fixed, num_steps=1),
            self._make_heads(glimpse_v_fixed, num_steps=1),
            logit_k_fixed.contiguous(),
        )

    def _get_log_p(self, fixed: _AttentionModelFixed, mask: torch.Tensor):
        query = fixed.context_node_projected[:, None, :]
        glimpse_k, glimpse_v, logit_k = fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key
        log_p, _ = self._one_to_many_logits(query, glimpse_k, glimpse_v, logit_k, mask)
        log_p = torch.log_softmax(log_p / 1.0, dim=-1)
        assert not torch.isnan(log_p).any()
        return log_p.exp(), mask

    def _one_to_many_logits(
        self,
        query: torch.Tensor,
        glimpse_k: torch.Tensor,
        glimpse_v: torch.Tensor,
        logit_k: torch.Tensor,
        mask: torch.Tensor,
    ):
        batch_size, num_steps, embed_dim = query.size()
        key_size = embed_dim // self.n_heads

        glimpse_q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_q, glimpse_k.transpose(-2, -1)) / math.sqrt(glimpse_q.size(-1))
        logits = compatibility.reshape([-1, 1, compatibility.shape[-1]])

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        # Slice down to leaf-node logits only.
        logits = logits[
            :, 0, self.internal_node_holder : self.internal_node_holder + self.leaf_node_holder
        ]
        if self.mask_logits:
            logits = logits.masked_fill(mask.bool(), float("-inf"))
        return logits, None

    def _make_heads(self, v: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )
