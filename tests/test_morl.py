"""MORL: reward vector + model + trainer + agent smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ----- reward vector -----


def test_compute_reward_vector_basic(container_40hc, eur_pallets_10):
    from app.algorithms.pct.morl_rewards import (
        N_OBJECTIVES,
        RewardComponents,
        compute_reward_vector,
    )
    from app.constraints.cog import CoGTracker
    from app.schemas import Dimensions, Placement, Position, Rotation

    item = eur_pallets_10[0]
    placement = Placement(
        item_id=item.id,
        position=Position(x_mm=0, y_mm=0, z_mm=0),
        rotation=Rotation.LWH,
        rotated_dimensions=Dimensions(length_mm=1200, width_mm=800, height_mm=1200),
    )
    cog_before = CoGTracker(container=container_40hc)
    cog_after = CoGTracker(container=container_40hc)
    cog_after.add(placement, item.weight_kg)

    rc = compute_reward_vector(
        new_placement=placement,
        new_item=item,
        container=container_40hc,
        prior_placements=[],
        items_by_id={item.id: item},
        cog_before=cog_before,
        cog_after=cog_after,
    )
    assert isinstance(rc, RewardComponents)
    v = rc.to_vector()
    assert len(v) == N_OBJECTIVES
    # First placement at the door corner: should get util > 0, AE = 1, stability = 1, CoG ≈ 1, LIFO = 1
    assert v[0] > 0
    assert v[1] == 1.0
    assert v[2] == 1.0
    assert v[4] == 1.0


def test_scalarise_matches_dot_product():
    from app.algorithms.pct.morl_rewards import scalarise
    pref = [0.5, 0.25, 0.125, 0.0625, 0.0625]
    rew = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = sum(p * r for p, r in zip(pref, rew))
    assert scalarise(rew, pref) == pytest.approx(expected)


# ----- model -----


def test_morl_drl_gat_forward_with_preference():
    from app.algorithms.pct.morl_model import MORL_DRL_GAT, PCTMORLConfig

    cfg = PCTMORLConfig(
        embedding_size=32, hidden_size=64, gat_layer_num=1,
        internal_node_holder=8, leaf_node_holder=4, internal_node_length=6,
        n_objectives=5,
    )
    model = MORL_DRL_GAT(cfg)
    B = 3
    graph_size = cfg.internal_node_holder + cfg.leaf_node_holder + 1
    max_dim = max(cfg.internal_node_length, 8, 6) + 2
    obs = torch.zeros(B, graph_size, max_dim)
    obs[:, cfg.internal_node_holder : cfg.internal_node_holder + 3, 8] = 1
    obs[:, : cfg.internal_node_holder + 3, -1] = 1
    obs[:, -1, -1] = 1
    pref = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.0, 0.5, 0.5, 0.0, 0.0],
    ])

    log_prob, action, entropy, value_vec = model(obs, pref)
    assert log_prob.shape == (B, 1)
    assert action.shape == (B, 1)
    assert value_vec.shape == (B, cfg.n_objectives)


def test_morl_trainer_one_iter(container_40hc, eur_pallets_10):
    from app.algorithms.pct.morl_model import MORL_DRL_GAT, PCTMORLConfig
    from app.algorithms.pct.morl_trainer import MORLPCTPPOTrainer, MORLPPOConfig
    from app.algorithms.pct.pct_env import PCTEnvConfig

    env_cfg = PCTEnvConfig(internal_node_holder=8, leaf_node_holder=4, max_candidates=4)
    model_cfg = PCTMORLConfig(
        embedding_size=16, hidden_size=32, gat_layer_num=1,
        internal_node_holder=env_cfg.internal_node_holder,
        leaf_node_holder=env_cfg.leaf_node_holder,
        internal_node_length=env_cfg.internal_node_length,
    )
    model = MORL_DRL_GAT(model_cfg)
    trainer = MORLPCTPPOTrainer(
        model,
        sample_voyage_fn=lambda: (container_40hc, eur_pallets_10),
        env_cfg=env_cfg,
        cfg=MORLPPOConfig(
            n_envs=2, rollout_steps=4, n_epochs=1, minibatch_size=4,
            log_every=1, autosave_every=999,
        ),
    )
    logs = []
    steps = trainer.train(total_steps=8, on_log=logs.append)
    assert steps >= 8
    assert len(logs) >= 1
    assert "mean_reward_vec" in logs[0]
    assert len(logs[0]["mean_reward_vec"]) == 5


def test_morl_save_load_roundtrip(tmp_path, container_40hc, eur_pallets_10):
    from app.algorithms.pct.morl_model import MORL_DRL_GAT, PCTMORLConfig
    from app.algorithms.pct.morl_trainer import MORLPCTPPOTrainer, MORLPPOConfig
    from app.algorithms.pct.pct_env import PCTEnvConfig

    env_cfg = PCTEnvConfig(internal_node_holder=8, leaf_node_holder=4, max_candidates=4)
    model_cfg = PCTMORLConfig(
        embedding_size=16, hidden_size=32, gat_layer_num=1,
        internal_node_holder=env_cfg.internal_node_holder,
        leaf_node_holder=env_cfg.leaf_node_holder,
        internal_node_length=env_cfg.internal_node_length,
    )
    model = MORL_DRL_GAT(model_cfg)
    trainer = MORLPCTPPOTrainer(
        model, sample_voyage_fn=lambda: (container_40hc, eur_pallets_10),
        env_cfg=env_cfg, cfg=MORLPPOConfig(n_envs=1, rollout_steps=2, n_epochs=1, minibatch_size=2),
    )
    trainer.train(total_steps=4)
    ckpt = tmp_path / "morl.pt"
    trainer.save(ckpt)

    # Build a fresh trainer and load
    model2 = MORL_DRL_GAT(model_cfg)
    trainer2 = MORLPCTPPOTrainer(
        model2, sample_voyage_fn=lambda: (container_40hc, eur_pallets_10),
        env_cfg=env_cfg, cfg=MORLPPOConfig(n_envs=1, rollout_steps=2),
    )
    steps = trainer2.load_checkpoint(ckpt)
    assert steps == trainer._global_steps


def test_morl_agent_select_with_preference(tmp_path, container_40hc, eur_pallets_10):
    from app.algorithms.base import solve
    from app.algorithms.pct.morl_agent import MORLPCTAgent
    from app.algorithms.pct.morl_model import MORL_DRL_GAT, PCTMORLConfig
    from app.algorithms.pct.morl_trainer import MORLPCTPPOTrainer, MORLPPOConfig
    from app.algorithms.pct.pct_env import PCTEnvConfig

    env_cfg = PCTEnvConfig(internal_node_holder=8, leaf_node_holder=4, max_candidates=4)
    model_cfg = PCTMORLConfig(
        embedding_size=16, hidden_size=32, gat_layer_num=1,
        internal_node_holder=env_cfg.internal_node_holder,
        leaf_node_holder=env_cfg.leaf_node_holder,
        internal_node_length=env_cfg.internal_node_length,
    )
    model = MORL_DRL_GAT(model_cfg)
    trainer = MORLPCTPPOTrainer(
        model, sample_voyage_fn=lambda: (container_40hc, eur_pallets_10),
        env_cfg=env_cfg, cfg=MORLPPOConfig(n_envs=1, rollout_steps=2, n_epochs=1, minibatch_size=2),
    )
    trainer.train(total_steps=4)
    ckpt = tmp_path / "morl_tiny.pt"
    trainer.save(ckpt)

    # Different preferences ⇒ different policies (in principle; here just verify it runs)
    for pref in [
        [1.0, 0.0, 0.0, 0.0, 0.0],     # util-only
        [0.0, 1.0, 0.0, 0.0, 0.0],     # access-only
        [0.2, 0.2, 0.2, 0.2, 0.2],     # balanced
    ]:
        agent = MORLPCTAgent(weights_path=ckpt, preference=pref)
        result, _ = solve(algorithm=agent, container=container_40hc, items=eur_pallets_10)
        assert len(result.placements) + len(result.unplaced_item_ids) == len(eur_pallets_10)


def test_morl_agent_rejects_invalid_preference(tmp_path, container_40hc, eur_pallets_10):
    """Validation: wrong shape, negative entries, or zero vector → ValueError."""
    from app.algorithms.pct.morl_agent import MORLPCTAgent
    from app.algorithms.pct.morl_model import MORL_DRL_GAT, PCTMORLConfig
    from app.algorithms.pct.morl_trainer import MORLPCTPPOTrainer, MORLPPOConfig
    from app.algorithms.pct.pct_env import PCTEnvConfig

    env_cfg = PCTEnvConfig(internal_node_holder=8, leaf_node_holder=4, max_candidates=4)
    model_cfg = PCTMORLConfig(
        embedding_size=16, hidden_size=32, gat_layer_num=1,
        internal_node_holder=env_cfg.internal_node_holder,
        leaf_node_holder=env_cfg.leaf_node_holder,
        internal_node_length=env_cfg.internal_node_length,
    )
    trainer = MORLPCTPPOTrainer(
        MORL_DRL_GAT(model_cfg),
        sample_voyage_fn=lambda: (container_40hc, eur_pallets_10),
        env_cfg=env_cfg, cfg=MORLPPOConfig(n_envs=1, rollout_steps=2),
    )
    trainer.train(total_steps=4)
    ckpt = tmp_path / "morl_tiny.pt"
    trainer.save(ckpt)

    with pytest.raises(ValueError):
        MORLPCTAgent(weights_path=ckpt, preference=[1.0, 0.0])           # wrong shape
    with pytest.raises(ValueError):
        MORLPCTAgent(weights_path=ckpt, preference=[-0.1, 0.5, 0.5, 0.1, 0.0])  # negative
    with pytest.raises(ValueError):
        MORLPCTAgent(weights_path=ckpt, preference=[0.0] * 5)            # zero
