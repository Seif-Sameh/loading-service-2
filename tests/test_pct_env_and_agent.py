"""Smoke tests for PCT env + agent + trainer."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_pct_env_observation_shape(container_40hc, eur_pallets_10):
    from app.algorithms.pct.pct_env import PCTEnv, PCTEnvConfig

    cfg = PCTEnvConfig(internal_node_holder=16, leaf_node_holder=8, max_candidates=8)
    env = PCTEnv(container=container_40hc, items=eur_pallets_10, cfg=cfg)
    obs, info = env.reset()

    expected = (16 + 8 + 1, max(cfg.internal_node_length, 8, 6) + 2)
    assert obs.shape == expected
    assert obs.dtype.name == "float32"
    assert info["n_remaining"] == 10
    # at reset there should be at least one feasible candidate
    assert info["n_candidates"] > 0
    # next-item slot should have its real-mask flag set
    next_row = cfg.internal_node_holder + cfg.leaf_node_holder
    assert obs[next_row, -1] == 1.0


def test_pct_env_step_advances(container_40hc, eur_pallets_10):
    from app.algorithms.pct.pct_env import PCTEnv, PCTEnvConfig

    env = PCTEnv(
        container=container_40hc,
        items=eur_pallets_10,
        cfg=PCTEnvConfig(internal_node_holder=16, leaf_node_holder=8, max_candidates=8),
    )
    env.reset()
    obs, reward, done, _, info = env.step(0)
    assert info["n_placed"] == 1
    assert info["n_remaining"] == 9
    assert reward != 0.0


def test_pct_trainer_one_iter_runs(container_40hc, eur_pallets_10):
    from app.algorithms.pct.pct_env import PCTEnvConfig
    from app.algorithms.pct.pct_model import DRL_GAT, PCTConfig
    from app.algorithms.pct.ppo_trainer import PCTPPOTrainer, PPOConfig

    env_cfg = PCTEnvConfig(internal_node_holder=8, leaf_node_holder=4, max_candidates=4)
    model_cfg = PCTConfig(
        embedding_size=16,
        hidden_size=32,
        gat_layer_num=1,
        internal_node_holder=env_cfg.internal_node_holder,
        leaf_node_holder=env_cfg.leaf_node_holder,
        internal_node_length=env_cfg.internal_node_length,
    )
    model = DRL_GAT(model_cfg)
    voyages = [(container_40hc, eur_pallets_10)]
    trainer = PCTPPOTrainer(
        model,
        sample_voyage_fn=lambda: voyages[0],
        env_cfg=env_cfg,
        cfg=PPOConfig(
            n_envs=2,
            rollout_steps=4,
            n_epochs=1,
            minibatch_size=4,
            log_every=1,
            autosave_every=999,
        ),
    )

    logs: list[dict] = []
    steps_done = trainer.train(total_steps=8, on_log=logs.append)
    assert steps_done >= 8
    assert len(logs) >= 1


def test_pct_trainer_save_and_resume_roundtrip(tmp_path, container_40hc, eur_pallets_10):
    from app.algorithms.pct.pct_env import PCTEnvConfig
    from app.algorithms.pct.pct_model import DRL_GAT, PCTConfig
    from app.algorithms.pct.ppo_trainer import PCTPPOTrainer, PPOConfig

    env_cfg = PCTEnvConfig(internal_node_holder=8, leaf_node_holder=4, max_candidates=4)
    model_cfg = PCTConfig(
        embedding_size=16,
        hidden_size=32,
        gat_layer_num=1,
        internal_node_holder=env_cfg.internal_node_holder,
        leaf_node_holder=env_cfg.leaf_node_holder,
        internal_node_length=env_cfg.internal_node_length,
    )
    model = DRL_GAT(model_cfg)
    trainer = PCTPPOTrainer(
        model,
        sample_voyage_fn=lambda: (container_40hc, eur_pallets_10),
        env_cfg=env_cfg,
        cfg=PPOConfig(n_envs=1, rollout_steps=2, n_epochs=1, minibatch_size=2),
    )
    trainer.train(total_steps=4, on_log=None)
    ckpt = tmp_path / "ckpt.pt"
    trainer.save(ckpt)
    assert ckpt.is_file()

    # Build a fresh trainer and load — global step counter should restore
    model2 = DRL_GAT(model_cfg)
    trainer2 = PCTPPOTrainer(
        model2,
        sample_voyage_fn=lambda: (container_40hc, eur_pallets_10),
        env_cfg=env_cfg,
        cfg=PPOConfig(n_envs=1, rollout_steps=2),
    )
    steps = trainer2.load_checkpoint(ckpt)
    assert steps == trainer._global_steps


def test_pct_agent_runs_after_short_training(tmp_path, container_40hc, eur_pallets_10):
    from app.algorithms.base import solve
    from app.algorithms.pct.pct_agent import PCTPackingAgent
    from app.algorithms.pct.pct_env import PCTEnvConfig
    from app.algorithms.pct.pct_model import DRL_GAT, PCTConfig
    from app.algorithms.pct.ppo_trainer import PCTPPOTrainer, PPOConfig

    env_cfg = PCTEnvConfig(internal_node_holder=8, leaf_node_holder=4, max_candidates=4)
    model_cfg = PCTConfig(
        embedding_size=16,
        hidden_size=32,
        gat_layer_num=1,
        internal_node_holder=env_cfg.internal_node_holder,
        leaf_node_holder=env_cfg.leaf_node_holder,
        internal_node_length=env_cfg.internal_node_length,
    )
    model = DRL_GAT(model_cfg)
    trainer = PCTPPOTrainer(
        model,
        sample_voyage_fn=lambda: (container_40hc, eur_pallets_10),
        env_cfg=env_cfg,
        cfg=PPOConfig(n_envs=1, rollout_steps=2, n_epochs=1, minibatch_size=2),
    )
    trainer.train(total_steps=4, on_log=None)
    ckpt = tmp_path / "tiny_pct.pt"
    trainer.save(ckpt)

    agent = PCTPackingAgent(weights_path=ckpt)
    result, _ = solve(algorithm=agent, container=container_40hc, items=eur_pallets_10)
    # Untrained agent might place zero items, but it MUST produce a valid SolveResult
    assert result.placements is not None
    assert len(result.placements) + len(result.unplaced_item_ids) == len(eur_pallets_10)
