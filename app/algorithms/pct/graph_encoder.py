"""Multi-head graph attention encoder for PCT.

Ported with light cleanups from
https://github.com/alexfrom0815/Online-3D-BPP-PCT/blob/master/graph_encoder.py.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class _SkipConnection(nn.Module):
    """Residual wrapper that preserves the dict-shaped input PCT uses."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, data):
        return {
            "data": data["data"] + self.module(data),
            "mask": data["mask"],
            "graph_size": data["graph_size"],
            "evaluate": data.get("evaluate", False),
        }


class _SkipConnectionLinear(nn.Module):
    """Variant of skip-connection that feeds only the data tensor through ``module``."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, data):
        return {
            "data": data["data"] + self.module(data["data"]),
            "mask": data["mask"],
            "graph_size": data["graph_size"],
            "evaluate": data.get("evaluate", False),
        }


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with mask handling for padded graph nodes."""

    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        embed_dim: int | None = None,
        val_dim: int | None = None,
        key_dim: int | None = None,
    ) -> None:
        super().__init__()
        if val_dim is None:
            assert embed_dim is not None
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1.0 / math.sqrt(key_dim)

        self.W_query = nn.Linear(input_dim, key_dim, bias=False)
        self.W_key = nn.Linear(input_dim, key_dim, bias=False)
        self.W_val = nn.Linear(input_dim, val_dim, bias=False)
        self.W_out = nn.Linear(key_dim, embed_dim) if embed_dim is not None else None

        self._init_parameters()

    def _init_parameters(self) -> None:
        for p in self.parameters():
            stdv = 1.0 / math.sqrt(p.size(-1))
            p.data.uniform_(-stdv, stdv)

    def forward(self, data: dict, h: torch.Tensor | None = None) -> torch.Tensor:
        q = data["data"]
        mask = data["mask"]
        graph_size = data["graph_size"]
        evaluate = data.get("evaluate", False)

        if h is None:
            h = q

        batch_size = int(q.size(0) / graph_size)
        n_query = graph_size
        input_dim = h.size(-1)
        assert input_dim == self.input_dim

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = self.W_query(qflat).view(shp_q)
        K = self.W_key(hflat).view(shp)
        V = self.W_val(hflat).view(shp)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # mask broadcasting: (B, graph_size) -> (1, B, n_query, graph_size)
        mask_b = mask.unsqueeze(1).repeat((1, graph_size, 1)).bool()
        mask_full = mask_b.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
        compatibility = compatibility.masked_fill(mask_full, float("-inf") if evaluate else -30.0)

        attn = torch.softmax(compatibility, dim=-1)

        # Hard-zero entries that softmax produced under fully-masked rows (avoids NaN propagation)
        attn = attn.masked_fill(mask_full, 0.0)

        heads = torch.matmul(attn, V)
        out = self.W_out(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.n_heads * self.val_dim)
        ).view(batch_size * n_query, self.embed_dim)
        return out


class _MultiHeadAttentionLayer(nn.Sequential):
    def __init__(self, n_heads: int, embed_dim: int, ff_hidden: int = 128) -> None:
        super().__init__(
            _SkipConnection(MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)),
            _SkipConnectionLinear(
                nn.Sequential(
                    nn.Linear(embed_dim, ff_hidden),
                    nn.ReLU(),
                    nn.Linear(ff_hidden, embed_dim),
                )
                if ff_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
        )


class GraphAttentionEncoder(nn.Module):
    """Stacks ``n_layers`` graph-attention blocks over a dict-shaped input."""

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        node_dim: int | None = None,
        ff_hidden: int = 128,
        graph_size: int | None = None,
    ) -> None:
        super().__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.graph_size = graph_size
        self.layers = nn.Sequential(
            *(_MultiHeadAttentionLayer(n_heads, embed_dim, ff_hidden) for _ in range(n_layers))
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        evaluate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.init_embed is not None:
            h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
        else:
            h = x

        data = {"data": h, "mask": mask, "graph_size": self.graph_size, "evaluate": evaluate}
        h = self.layers(data)["data"]
        # Per-graph mean pool (used by callers that want a global context vector).
        graph_mean = h.view(int(h.size(0) / self.graph_size), self.graph_size, -1).mean(dim=1)
        return h, graph_mean
