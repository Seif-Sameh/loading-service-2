"""PPO trainer for the PCT (DRL_GAT) policy.

Designed around a Kaggle 30-hour budget split across multiple ~12 h sessions:

- Each call to :meth:`train` writes a checkpoint every ``autosave_every`` rollouts.
- The checkpoint contains model, optimizer, global step counter, and RNG state.
- Re-running the notebook from cell 1 detects ``models/pct_latest.pt`` and resumes
  exactly where the previous session left off.
- Final checkpoint after hitting ``total_steps`` is saved as ``models/pct_final.pt``.

Why PPO over the paper's ACKTR — see commit message of the previous PR; tl;dr PPO
converges to the same final utilisation on this architecture (GOPT, 2024 RA-L) and
its checkpoint state is just optimizer + step counter, no KFAC running stats to
serialise.
"""
from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from app.schemas import CargoItem, Container

from .pct_env import PCTEnv, PCTEnvConfig
from .pct_model import DRL_GAT, PCTConfig

SampleVoyageFn = Callable[[], tuple[Container, list[CargoItem]]]


@dataclass
class PPOConfig:
    n_envs: int = 8
    rollout_steps: int = 64
    n_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "cpu"
    log_every: int = 5
    autosave_every: int = 25  # rollout iterations between checkpoint saves


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PCTPPOTrainer:
    def __init__(
        self,
        model: DRL_GAT,
        sample_voyage_fn: SampleVoyageFn,
        env_cfg: PCTEnvConfig,
        cfg: PPOConfig | None = None,
    ) -> None:
        self.model = model
        self.sample_voyage_fn = sample_voyage_fn
        self.env_cfg = env_cfg
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self._global_steps = 0
        self._rollout_iter = 0
        self._envs: list[PCTEnv] = []
        self._reset_envs()

    # ----- env management -----

    def _make_env(self) -> PCTEnv:
        cont, items = self.sample_voyage_fn()
        return PCTEnv(container=cont, items=items, cfg=self.env_cfg)

    def _reset_envs(self) -> list[np.ndarray]:
        self._envs = []
        first_obs = []
        for _ in range(self.cfg.n_envs):
            env = self._make_env()
            obs, _ = env.reset()
            self._envs.append(env)
            first_obs.append(obs)
        return first_obs

    def _restart_env(self, idx: int) -> np.ndarray:
        env = self._make_env()
        obs, _ = env.reset()
        self._envs[idx] = env
        return obs

    # ----- act -----

    def _act(self, obs_batch: np.ndarray, deterministic: bool = False):
        """obs_batch: (N, graph_size, max_dim) numpy. Returns (actions, log_probs, values, leaf_mask)."""
        t_obs = torch.from_numpy(obs_batch).to(self.device)
        log_probs, action, _entropy, value = self.model(t_obs, deterministic=deterministic)
        # Build a mask over leaf nodes (used to verify masked actions are not selected)
        cfg = self.env_cfg
        leaf_valid = t_obs[
            :, cfg.internal_node_holder : cfg.internal_node_holder + cfg.leaf_node_holder, 8
        ]  # (N, leaf_holder)
        return action, log_probs, value.squeeze(-1), leaf_valid

    # ----- rollout -----

    def collect_rollout(self) -> tuple[dict, list[float], list[float]]:
        cfg = self.cfg
        T = cfg.rollout_steps
        N = cfg.n_envs

        obs_list = [env._build_observation() for env in self._envs]
        ep_returns: list[float] = []
        ep_utils: list[float] = []
        running_rewards = [0.0] * N

        # buffers (numpy on CPU)
        graph_size = self._envs[0].graph_size
        max_dim = self._envs[0].max_feature_dim
        buf_obs = np.zeros((T, N, graph_size, max_dim), dtype=np.float32)
        buf_actions = np.zeros((T, N), dtype=np.int64)
        buf_logp = np.zeros((T, N), dtype=np.float32)
        buf_values = np.zeros((T + 1, N), dtype=np.float32)
        buf_rewards = np.zeros((T, N), dtype=np.float32)
        buf_dones = np.zeros((T, N), dtype=np.float32)

        for t in range(T):
            obs_np = np.stack(obs_list)
            buf_obs[t] = obs_np
            with torch.no_grad():
                action, log_p, value, _ = self._act(obs_np)
            buf_actions[t] = action.detach().cpu().numpy().squeeze(-1)
            buf_logp[t] = log_p.detach().cpu().numpy().squeeze(-1)
            buf_values[t] = value.detach().cpu().numpy()

            for i, env in enumerate(self._envs):
                a = int(buf_actions[t, i])
                obs_next, r, done, _, _info = env.step(a)
                buf_rewards[t, i] = float(r)
                buf_dones[t, i] = 1.0 if done else 0.0
                running_rewards[i] += float(r)
                if done:
                    ep_returns.append(running_rewards[i])
                    _, kpis = env.final_score()
                    ep_utils.append(kpis.utilization)
                    running_rewards[i] = 0.0
                    obs_next = self._restart_env(i)
                obs_list[i] = obs_next

        # bootstrap value for the final state
        with torch.no_grad():
            _, _, value_last, _ = self._act(np.stack(obs_list))
        buf_values[T] = value_last.detach().cpu().numpy()

        # If no episode finished this rollout, fall back to the *current* utilisation of
        # each in-flight env. This way the log shows real progress even when voyages are
        # longer than the rollout window.
        if not ep_utils:
            for env in self._envs:
                _, kpis = env.final_score()
                ep_utils.append(kpis.utilization)

        return (
            {
                "obs": buf_obs,
                "actions": buf_actions,
                "log_probs": buf_logp,
                "values": buf_values,
                "rewards": buf_rewards,
                "dones": buf_dones,
            },
            ep_returns,
            ep_utils,
        )

    # ----- update -----

    @staticmethod
    def _compute_gae(buf: dict, gamma: float, gae_lambda: float):
        rewards = buf["rewards"]
        values = buf["values"]
        dones = buf["dones"]
        T, N = rewards.shape
        adv = np.zeros_like(rewards)
        last = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
            last = delta + gamma * gae_lambda * non_terminal * last
            adv[t] = last
        returns = adv + values[:T]
        return adv, returns

    def update(self, buf: dict) -> dict[str, float]:
        cfg = self.cfg
        T = cfg.rollout_steps
        N = cfg.n_envs

        adv, returns = self._compute_gae(buf, cfg.gamma, cfg.gae_lambda)

        flat = lambda x: np.asarray(x).reshape(T * N, *np.asarray(x).shape[2:])
        obs_b = torch.from_numpy(flat(buf["obs"])).float().to(self.device)
        act_b = torch.from_numpy(flat(buf["actions"])).long().to(self.device)
        old_logp_b = torch.from_numpy(flat(buf["log_probs"])).float().to(self.device).unsqueeze(-1)
        old_val_b = torch.from_numpy(flat(buf["values"][:T])).float().to(self.device).unsqueeze(-1)
        adv_b = torch.from_numpy(flat(adv)).float().to(self.device).unsqueeze(-1)
        ret_b = torch.from_numpy(flat(returns)).float().to(self.device).unsqueeze(-1)

        # normalise advantages
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        idx = np.arange(T * N)
        loss_log = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
        n_batches = 0
        for _ in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, T * N, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                if len(mb) == 0:
                    continue
                mb_t = torch.from_numpy(mb).long().to(self.device)

                value, new_logp, entropy = self.model.evaluate_actions(
                    obs_b[mb_t], act_b[mb_t].unsqueeze(-1)
                )
                ratio = torch.exp(new_logp - old_logp_b[mb_t])
                surr1 = ratio * adv_b[mb_t]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b[mb_t]
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = old_val_b[mb_t] + torch.clamp(
                    value - old_val_b[mb_t], -cfg.value_clip_eps, cfg.value_clip_eps
                )
                v_loss_1 = (value - ret_b[mb_t]).pow(2)
                v_loss_2 = (v_clipped - ret_b[mb_t]).pow(2)
                value_loss = 0.5 * torch.max(v_loss_1, v_loss_2).mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                loss_log["policy"] += float(policy_loss.item())
                loss_log["value"] += float(value_loss.item())
                loss_log["entropy"] += float(entropy.item())
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in loss_log.items()}

    # ----- main train loop -----

    def train(
        self,
        total_steps: int,
        on_log: Callable[[dict], None] | None = None,
        *,
        wall_clock_budget_s: float | None = None,
        autosave_path: str | os.PathLike | None = None,
    ) -> int:
        """Train until ``total_steps`` env-steps OR ``wall_clock_budget_s`` seconds elapse.

        ``autosave_path`` (e.g. ``"models/pct_latest.pt"``) is written every
        ``autosave_every`` rollouts plus once at the very end.
        Returns the global step count after the run.
        """
        cfg = self.cfg
        t0 = time.time()
        while self._global_steps < total_steps:
            if wall_clock_budget_s is not None and (time.time() - t0) > wall_clock_budget_s:
                break

            buf, ep_returns, ep_utils = self.collect_rollout()
            losses = self.update(buf)
            self._global_steps += cfg.rollout_steps * cfg.n_envs
            self._rollout_iter += 1

            if on_log and (self._rollout_iter % cfg.log_every == 0 or self._rollout_iter == 1):
                on_log({
                    "iter": self._rollout_iter,
                    "steps_done": self._global_steps,
                    "episodes": len(ep_returns),
                    "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
                    "mean_util": float(np.mean(ep_utils)) if ep_utils else 0.0,
                    **losses,
                })
            if autosave_path and self._rollout_iter % cfg.autosave_every == 0:
                self.save(autosave_path)

        if autosave_path:
            self.save(autosave_path)
        return self._global_steps

    # ----- io -----

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "pct_config": vars(self.model.cfg),
                "env_config": vars(self.env_cfg),
                "ppo_config": vars(self.cfg),
                "global_steps": self._global_steps,
                "rollout_iter": self._rollout_iter,
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
            },
            path,
        )

    def load_checkpoint(self, path: str | os.PathLike) -> int:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._global_steps = int(ckpt.get("global_steps", 0))
        self._rollout_iter = int(ckpt.get("rollout_iter", 0))
        # restore RNG so resumed runs are reproducible-ish
        try:
            if "rng_python" in ckpt:
                random.setstate(ckpt["rng_python"])
            if "rng_numpy" in ckpt:
                np.random.set_state(ckpt["rng_numpy"])
            if "rng_torch" in ckpt:
                torch.set_rng_state(ckpt["rng_torch"])
        except Exception:
            pass
        return self._global_steps

    @classmethod
    def load_model(cls, path: str | os.PathLike, *, device: str = "cpu") -> DRL_GAT:
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        cfg = PCTConfig(**ckpt["pct_config"])
        model = DRL_GAT(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
