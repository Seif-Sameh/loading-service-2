"""Init helpers + packed-observation slicing for PCT.

Adapted from https://github.com/alexfrom0815/Online-3D-BPP-PCT/blob/master/tools.py.
"""
from __future__ import annotations

import torch
from torch import nn


def init(module: nn.Module, weight_init, bias_init, gain: float = 1.0) -> nn.Module:
    """Apply orthogonal-style init to a Linear/Conv layer in-place."""
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def update_linear_schedule(optimizer, epoch: int, total: int, initial_lr: float) -> None:
    """Linear LR decay matching the PCT trainer."""
    lr = initial_lr - (initial_lr * (epoch / float(total)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def observation_decode_leaf_node(
    observation: torch.Tensor,
    internal_node_holder: int,
    internal_node_length: int,
    leaf_node_holder: int,
):
    """Slice a packed PCT observation tensor into its components.

    Layout (per batch element):
      observation[:, 0 : internal_node_holder, 0 : internal_node_length] = internal nodes
        — placed-item descriptors (position xyz, size xyz, optional density …)
      observation[:, internal_node_holder : internal_node_holder + leaf_node_holder, 0 : 8]
        — leaf nodes (candidate placements: position xyz, size xyz, orientation flags …)
      observation[:, internal_node_holder + leaf_node_holder :, 0 : 6]
        — current item descriptor (size xyz, …)
      observation[:, internal_node_holder : internal_node_holder + leaf_node_holder, 8]
        — leaf validity flag (1 = real candidate, 0 = padding)
      observation[:, :, -1]
        — global "real node" mask (1 where the slot holds a real graph node)

    Returns ``(internal_nodes, leaf_nodes, next_item, leaf_valid, full_mask)``.
    """
    internal_nodes = observation[:, 0:internal_node_holder, 0:internal_node_length]
    leaf_nodes = observation[
        :, internal_node_holder : internal_node_holder + leaf_node_holder, 0:8
    ]
    current_box = observation[:, internal_node_holder + leaf_node_holder :, 0:6]
    leaf_valid = observation[
        :, internal_node_holder : internal_node_holder + leaf_node_holder, 8
    ]
    full_mask = observation[:, :, -1]
    return internal_nodes, leaf_nodes, current_box, leaf_valid, full_mask
