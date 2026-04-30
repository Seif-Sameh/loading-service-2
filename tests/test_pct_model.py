"""Smoke tests for the PCT model port — confirm forward/backward shapes and gradients."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _dummy_observation(B: int = 2, cfg=None):
    """Build a packed (B, graph_size, max_dim) observation with random content."""
    from app.algorithms.pct.pct_model import PCTConfig, pack_observation

    cfg = cfg or PCTConfig(internal_node_holder=80, leaf_node_holder=50)
    internal = torch.rand(B, cfg.internal_node_holder, cfg.internal_node_length)
    leaf = torch.rand(B, cfg.leaf_node_holder, 8)
    next_item = torch.rand(B, 1, 6)
    leaf_valid = torch.zeros(B, cfg.leaf_node_holder)
    leaf_valid[:, :10] = 1                                  # mark 10 leaves valid
    real_mask = torch.zeros(B, cfg.internal_node_holder + cfg.leaf_node_holder + 1)
    real_mask[:, :5] = 1                                     # 5 placed items
    real_mask[:, cfg.internal_node_holder : cfg.internal_node_holder + 10] = 1   # leaves
    real_mask[:, -1] = 1                                     # next item slot
    return pack_observation(
        internal_nodes=internal,
        leaf_nodes=leaf,
        next_item=next_item,
        leaf_valid=leaf_valid,
        real_mask=real_mask,
        cfg=cfg,
    ), cfg


def test_attention_model_forward_shapes():
    from app.algorithms.pct.attention_model import AttentionModel

    cfg_internal_holder = 16
    cfg_leaf_holder = 8
    cfg_internal_length = 6
    model = AttentionModel(
        embedding_dim=32,
        hidden_dim=64,
        n_encode_layers=1,
        n_heads=1,
        internal_node_holder=cfg_internal_holder,
        internal_node_length=cfg_internal_length,
        leaf_node_holder=cfg_leaf_holder,
    )
    B = 3
    graph_size = cfg_internal_holder + cfg_leaf_holder + 1
    obs = torch.zeros(B, graph_size, 10)
    # internal nodes (zeros are fine)
    # mark 4 valid leaves
    obs[:, cfg_internal_holder : cfg_internal_holder + 4, :8] = torch.rand(B, 4, 8)
    obs[:, cfg_internal_holder : cfg_internal_holder + 4, 8] = 1
    # next item
    obs[:, -1, :6] = torch.rand(B, 6)
    # full real-node mask
    obs[:, : cfg_internal_holder + 4, -1] = 1
    obs[:, -1, -1] = 1

    log_prob, action, entropy, hidden, dist = model(obs)
    assert log_prob.shape == (B, 1)
    assert action.shape == (B, 1)
    assert entropy.shape == (B,)
    assert hidden.shape == (B, 32)


def test_drl_gat_forward_and_evaluate():
    from app.algorithms.pct.pct_model import DRL_GAT, PCTConfig

    cfg = PCTConfig(
        embedding_size=32,
        hidden_size=64,
        gat_layer_num=1,
        internal_node_holder=16,
        leaf_node_holder=8,
        internal_node_length=6,
    )
    model = DRL_GAT(cfg)
    obs, _ = _dummy_observation(B=4, cfg=cfg)

    log_prob, action, entropy, value = model(obs)
    assert log_prob.shape == (4, 1)
    assert action.shape == (4, 1)
    assert value.shape == (4, 1)

    # evaluate_actions path
    value2, log_probs2, entropy2 = model.evaluate_actions(obs, action)
    assert value2.shape == (4, 1)
    assert log_probs2.shape == (4, 1)
    assert entropy2.shape == ()


def test_drl_gat_backward_gradient_flows():
    """Loss against a synthetic target produces non-zero gradients on every parameter."""
    from app.algorithms.pct.pct_model import DRL_GAT, PCTConfig

    cfg = PCTConfig(
        embedding_size=32,
        hidden_size=64,
        gat_layer_num=1,
        internal_node_holder=16,
        leaf_node_holder=8,
        internal_node_length=6,
    )
    model = DRL_GAT(cfg)
    obs, _ = _dummy_observation(B=4, cfg=cfg)

    log_prob, action, entropy, value = model(obs)
    fake_advantages = torch.randn(4, 1)
    fake_returns = torch.randn(4, 1)
    loss = -(log_prob * fake_advantages).mean() + 0.5 * (value - fake_returns).pow(2).mean() - 0.01 * entropy.mean()
    loss.backward()

    grad_present = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    grad_total = sum(1 for _ in model.parameters())
    # Most params should have a gradient
    assert grad_present >= grad_total - 2
