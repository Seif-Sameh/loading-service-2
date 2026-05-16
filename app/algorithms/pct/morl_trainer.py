"""MORL PPO trainer for the preference-conditioned PCT policy.

Architectural changes vs the single-objective PPO trainer:

1. Each episode samples a preference vector w ~ Dirichlet(α) on the simplex.
2. The env step returns a *vector* reward; the trainer keeps both the vector and
   the scalarised reward w · r.
3. The critic produces n_objectives values; advantages are computed per-objective
   then scalarised with the same w for the PPO update.
4. Save / load checkpoints include the n_objectives so eval-time PCTMORLAgent
   can rebuild the matching network shape.

Warm-start from a single-objective PCT checkpoint is supported: parameters that
exist in both models (the GAT encoder, the pointer head, the input MLPs) are
loaded by name; the new ``pref_embed`` MLP and the per-objective critic are
randomly initialised.
"""
from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from app.schemas import CargoItem, Container

from .morl_model import MORL_DRL_GAT, N_OBJECTIVES, PCTMORLConfig
from .morl_rewards import RewardComponents, compute_reward_vector, scalarise
from .pct_env import PCTEnv, PCTEnvConfig

SampleVoyageFn = Callable[[], tuple[Container, list[CargoItem]]]


@dataclass
class MORLPPOConfig:
    n_envs: int = 8
    rollout_steps: int = 64
    n_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.03           # bumped from 0.01 to prevent preference collapse
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "cpu"
    log_every: int = 5
    autosave_every: int = 25
    n_objectives: int = N_OBJECTIVES
    # Dirichlet concentration α.
    #   α = 1.0 ⇒ uniform on the simplex (balanced preferences in expectation)
    #   α < 1   ⇒ CORNER-BIASED (most samples concentrate one objective). Recommended.
    #   α > 1   ⇒ centre-biased (most samples are near-balanced; rarely a single-objective extreme)
    # We default to 0.5 so the trainer sees plenty of "this episode is about util" /
    # "this episode is about LIFO" extremes — without those the policy can't learn
    # preference-conditional behaviour.
    dirichlet_alpha: float = 0.5


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class MORLPCTPPOTrainer:
    def __init__(
        self,
        model: MORL_DRL_GAT,
        sample_voyage_fn: SampleVoyageFn,
        env_cfg: PCTEnvConfig,
        cfg: MORLPPOConfig | None = None,
    ) -> None:
        self.model = model
        self.sample_voyage_fn = sample_voyage_fn
        self.env_cfg = env_cfg
        self.cfg = cfg or MORLPPOConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self._global_steps = 0
        self._rollout_iter = 0
        self._envs: list[PCTEnv] = []
        self._env_prefs: list[np.ndarray] = []           # one preference per env
        self._env_cog_before: list = []                  # CoG state snapshot for reward calc
        self._reset_envs()

    # ----- preference sampling -----

    def _sample_preference(self) -> np.ndarray:
        """Sample one preference vector on the simplex."""
        alpha = np.full(self.cfg.n_objectives, self.cfg.dirichlet_alpha, dtype=np.float32)
        w = np.random.dirichlet(alpha).astype(np.float32)
        return w

    # ----- env management -----

    def _make_env(self) -> tuple[PCTEnv, np.ndarray]:
        cont, items = self.sample_voyage_fn()
        env = PCTEnv(container=cont, items=items, cfg=self.env_cfg)
        pref = self._sample_preference()
        return env, pref

    def _reset_envs(self) -> list[np.ndarray]:
        from app.constraints.cog import CoGTracker

        self._envs = []
        self._env_prefs = []
        self._env_cog_before = []
        first_obs = []
        for _ in range(self.cfg.n_envs):
            env, pref = self._make_env()
            obs, _ = env.reset()
            self._envs.append(env)
            self._env_prefs.append(pref)
            self._env_cog_before.append(CoGTracker(container=env.container))
            first_obs.append(obs)
        return first_obs

    def _restart_env(self, idx: int) -> np.ndarray:
        from app.constraints.cog import CoGTracker

        env, pref = self._make_env()
        obs, _ = env.reset()
        self._envs[idx] = env
        self._env_prefs[idx] = pref
        self._env_cog_before[idx] = CoGTracker(container=env.container)
        return obs

    # ----- act -----

    def _act(self, obs_batch: np.ndarray, pref_batch: np.ndarray):
        """obs: (N, graph, max_dim), pref: (N, n_obj)."""
        t_obs = torch.from_numpy(obs_batch).to(self.device)
        t_pref = torch.from_numpy(pref_batch).to(self.device)
        log_probs, action, _entropy, value_vec = self.model(t_obs, t_pref)
        return action, log_probs, value_vec

    # ----- rollout -----

    def collect_rollout(self) -> tuple[dict, list[float], list[float], list[list[float]]]:
        """Returns (buffers, ep_scalar_returns, ep_utils, ep_reward_vectors)."""
        cfg = self.cfg
        T = cfg.rollout_steps
        N = cfg.n_envs
        O = cfg.n_objectives

        obs_list = [env._build_observation() for env in self._envs]
        ep_scalar_returns: list[float] = []
        ep_utils: list[float] = []
        ep_reward_vectors: list[list[float]] = []
        running_scalar = [0.0] * N
        running_vec = [np.zeros(O, dtype=np.float32) for _ in range(N)]

        graph_size = self._envs[0].graph_size
        max_dim = self._envs[0].max_feature_dim
        buf_obs = np.zeros((T, N, graph_size, max_dim), dtype=np.float32)
        buf_prefs = np.zeros((T, N, O), dtype=np.float32)
        buf_actions = np.zeros((T, N), dtype=np.int64)
        buf_logp = np.zeros((T, N), dtype=np.float32)
        buf_values = np.zeros((T + 1, N, O), dtype=np.float32)  # per-objective values
        buf_rewards = np.zeros((T, N, O), dtype=np.float32)     # per-objective rewards
        buf_dones = np.zeros((T, N), dtype=np.float32)

        for t in range(T):
            obs_np = np.stack(obs_list)
            pref_np = np.stack(self._env_prefs)
            buf_obs[t] = obs_np
            buf_prefs[t] = pref_np
            with torch.no_grad():
                action, log_p, value_vec = self._act(obs_np, pref_np)
            buf_actions[t] = action.detach().cpu().numpy().squeeze(-1)
            buf_logp[t] = log_p.detach().cpu().numpy().squeeze(-1)
            buf_values[t] = value_vec.detach().cpu().numpy()

            for i, env in enumerate(self._envs):
                items_by_id = {it.id: it for it in env._inner.items}
                cog_before_snapshot = _copy_cog(self._env_cog_before[i])

                # remember which item is current BEFORE the step
                current_item = env._inner.state.current_item

                a = int(buf_actions[t, i])
                obs_next, _scalar_r, done, _, _info = env.step(a)

                # If the env actually placed something, compute the reward vector.
                if current_item is not None and env._inner.state.placements and (
                    env._inner.state.placements[-1].item_id == current_item.id
                ):
                    new_placement = env._inner.state.placements[-1]
                    prior_placements = env._inner.state.placements[:-1]
                    cog_after = _copy_cog(self._env_cog_before[i])
                    cog_after.add(new_placement, current_item.weight_kg)
                    self._env_cog_before[i] = cog_after  # next step's "before"

                    rc: RewardComponents = compute_reward_vector(
                        new_placement=new_placement,
                        new_item=current_item,
                        container=env.container,
                        prior_placements=prior_placements,
                        items_by_id=items_by_id,
                        cog_before=cog_before_snapshot,
                        cog_after=cog_after,
                    )
                    r_vec = np.asarray(rc.to_vector(), dtype=np.float32)
                else:
                    # Step didn't actually place (e.g. invalid action absorbed). Zero reward.
                    r_vec = np.zeros(O, dtype=np.float32)

                scalar_r = float(scalarise(r_vec.tolist(), self._env_prefs[i].tolist()))
                buf_rewards[t, i] = r_vec
                buf_dones[t, i] = 1.0 if done else 0.0
                running_scalar[i] += scalar_r
                running_vec[i] += r_vec

                if done:
                    ep_scalar_returns.append(running_scalar[i])
                    ep_reward_vectors.append(running_vec[i].tolist())
                    _, kpis = env.final_score()
                    ep_utils.append(kpis.utilization)
                    running_scalar[i] = 0.0
                    running_vec[i] = np.zeros(O, dtype=np.float32)
                    obs_next = self._restart_env(i)

                obs_list[i] = obs_next

        # bootstrap value for the last observation
        with torch.no_grad():
            _, _, value_last = self._act(np.stack(obs_list), np.stack(self._env_prefs))
        buf_values[T] = value_last.detach().cpu().numpy()

        if not ep_utils:
            # No episode finished — fall back to running util so the log isn't empty.
            for env in self._envs:
                _, kpis = env.final_score()
                ep_utils.append(kpis.utilization)

        return (
            {
                "obs":      buf_obs,
                "prefs":    buf_prefs,
                "actions":  buf_actions,
                "log_probs": buf_logp,
                "values":   buf_values,
                "rewards":  buf_rewards,
                "dones":    buf_dones,
            },
            ep_scalar_returns,
            ep_utils,
            ep_reward_vectors,
        )

    # ----- update -----

    @staticmethod
    def _compute_gae(buf: dict, gamma: float, gae_lambda: float):
        """GAE per-objective. Returns (advantages, returns) both (T, N, O)."""
        rewards = buf["rewards"]      # (T, N, O)
        values  = buf["values"]       # (T+1, N, O)
        dones   = buf["dones"][..., None]  # (T, N, 1) broadcasts over O
        T, N, O = rewards.shape
        adv = np.zeros_like(rewards)
        last = np.zeros((N, O), dtype=np.float32)
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
        O = cfg.n_objectives

        adv, returns = self._compute_gae(buf, cfg.gamma, cfg.gae_lambda)

        flat = lambda x: np.asarray(x).reshape(T * N, *np.asarray(x).shape[2:])
        obs_b   = torch.from_numpy(flat(buf["obs"])).float().to(self.device)
        pref_b  = torch.from_numpy(flat(buf["prefs"])).float().to(self.device)
        act_b   = torch.from_numpy(flat(buf["actions"])).long().to(self.device)
        old_logp_b = torch.from_numpy(flat(buf["log_probs"])).float().to(self.device).unsqueeze(-1)
        old_val_b  = torch.from_numpy(flat(buf["values"][:T])).float().to(self.device)  # (TN, O)
        adv_b      = torch.from_numpy(flat(adv)).float().to(self.device)                # (TN, O)
        ret_b      = torch.from_numpy(flat(returns)).float().to(self.device)            # (TN, O)

        # Scalarise per-sample using that sample's preference
        scalar_adv = (pref_b * adv_b).sum(dim=-1, keepdim=True)
        # Normalise scalar advantage
        scalar_adv = (scalar_adv - scalar_adv.mean()) / (scalar_adv.std() + 1e-8)

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

                value_vec, new_logp, entropy = self.model.evaluate_actions(
                    obs_b[mb_t], pref_b[mb_t], act_b[mb_t].unsqueeze(-1),
                )
                ratio = torch.exp(new_logp - old_logp_b[mb_t])
                surr1 = ratio * scalar_adv[mb_t]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * scalar_adv[mb_t]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Per-objective value loss, then sum.
                v_clipped = old_val_b[mb_t] + torch.clamp(
                    value_vec - old_val_b[mb_t], -cfg.value_clip_eps, cfg.value_clip_eps
                )
                v_loss_1 = (value_vec - ret_b[mb_t]).pow(2)
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

    # ----- public train loop -----

    def train(
        self,
        total_steps: int,
        on_log: Callable[[dict], None] | None = None,
        *,
        wall_clock_budget_s: float | None = None,
        autosave_path: str | os.PathLike | None = None,
    ) -> int:
        cfg = self.cfg
        t0 = time.time()
        while self._global_steps < total_steps:
            if wall_clock_budget_s is not None and (time.time() - t0) > wall_clock_budget_s:
                break

            buf, ep_returns, ep_utils, ep_vecs = self.collect_rollout()
            losses = self.update(buf)
            self._global_steps += cfg.rollout_steps * cfg.n_envs
            self._rollout_iter += 1

            if on_log and (self._rollout_iter % cfg.log_every == 0 or self._rollout_iter == 1):
                mean_vec = np.mean(ep_vecs, axis=0).tolist() if ep_vecs else [0.0] * cfg.n_objectives
                on_log({
                    "iter": self._rollout_iter,
                    "steps_done": self._global_steps,
                    "episodes": len(ep_returns),
                    "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
                    "mean_util": float(np.mean(ep_utils)) if ep_utils else 0.0,
                    "mean_reward_vec": mean_vec,
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
                "morl": True,
            },
            path,
        )

    def load_checkpoint(self, path: str | os.PathLike, *, strict: bool = True) -> int:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        sd = ckpt["model_state"]
        # If warm-starting from a single-objective PCT checkpoint, the pref_embed and
        # the per-objective critic won't exist. Load the shared parameters non-strictly.
        if "morl" not in ckpt:
            strict = False
        missing, unexpected = self.model.load_state_dict(sd, strict=strict)
        if missing or unexpected:
            print(f"warm-start: missing={len(missing)} keys, unexpected={len(unexpected)} keys (expected when loading single-objective PCT)")
        if "optimizer_state" in ckpt and ckpt.get("morl"):
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception:
                pass
        self._global_steps = int(ckpt.get("global_steps", 0))
        self._rollout_iter = int(ckpt.get("rollout_iter", 0))
        return self._global_steps


def _copy_cog(src):
    """Shallow-clone a CoGTracker so we can snapshot it before a placement."""
    from app.constraints.cog import CoGTracker

    new = CoGTracker(container=src.container)
    new.total_weight_kg = src.total_weight_kg
    new._wx_sum = src._wx_sum
    new._wy_sum = src._wy_sum
    new._wz_sum = src._wz_sum
    return new
