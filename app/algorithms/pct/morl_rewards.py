"""Multi-objective reward computation for 3D bin packing.

Decomposes the scalar packing reward into a 5-vector covering operationally-relevant
objectives. Each component is normalised to roughly the same magnitude so the
preference-vector dot product is meaningful.

Objectives (in fixed order — the preference vector indexes match this list):

  0. util_gain         — volume of the placed item / container volume.
                         Per-step value: 0.00–0.05; cumulative over voyage: 0.3–0.8.
  1. access_eff_gain   — 1 if the new placement's centroid lies in the first
                         ``door_zone_fraction`` (default 20 %) of container length,
                         else 0. Operationally: how LIFO-friendly is the load.
  2. stability_gain    — 1 if the new placement's support ratio ≥ ``stable_threshold``
                         (default 0.7), else 0. Discourages overhanging stacks.
  3. cog_balance_gain  — 1 − (|long_dev| + |lat_dev|) clipped to [-1, 1].
                         The CoG deviations are already in [-0.5, 0.5]; subtracting
                         their absolute sum from 1 rewards balanced packings.
  4. lifo_gain         — 1 if the new placement does not create a LIFO violation
                         relative to existing placements, else 0.

These are designed to be summed across the episode (PPO sums step rewards anyway) and
the cumulative vector has each component in roughly [0, n_steps], which after the
preference scalarisation `w · r` produces a stable training signal.

References:
- Yang, Sun, Narasimhan (NeurIPS 2019) — *A Generalized Algorithm for MORL*
- Hayes et al. (AAMAS 2022) — *A Practical Guide to MORL*
- Multi-objective 3D BPP via Meta-RL (2025) — for the choice to use vector returns
"""
from __future__ import annotations

from dataclasses import dataclass

from app.constraints.cog import CoGTracker
from app.schemas import CargoItem, Container, Placement

N_OBJECTIVES = 5

# Per-component scaling so each step's reward contribution is roughly the same magnitude.
# Diagnostic at end of Day 1: util_gain raw is ~0.001-0.01 per step (volume / container volume),
# while access_eff / stability / lifo are 0/1 per step. Without rescaling, the scalarised
# reward signal under preference w=(1,0,0,0,0) is 100-400x smaller than under w=(0,0,0,0,1),
# and the policy collapses to ignoring the preference vector and maximising whichever
# component is easiest. Multiplying util_gain by ~100 brings it in line.
DEFAULT_SCALE = (
    100.0,   # util_gain        (raw ~0.001-0.01 per step  -> ~0.1-1.0 after scale)
    1.0,     # access_eff_gain  (raw 0 or 1)
    1.0,     # stability_gain   (raw 0 or 1)
    1.0,     # cog_balance_gain (raw 0.95-1.0; small drift per step)
    1.0,     # lifo_gain        (raw 0 or 1)
)


@dataclass(frozen=True)
class RewardComponents:
    """One step's worth of reward, broken out for inspection."""

    util_gain: float
    access_eff_gain: float
    stability_gain: float
    cog_balance_gain: float
    lifo_gain: float

    def to_vector(self) -> list[float]:
        return [
            self.util_gain,
            self.access_eff_gain,
            self.stability_gain,
            self.cog_balance_gain,
            self.lifo_gain,
        ]


def _support_ratio_from_placement(
    placement: Placement,
    prior_placements: list[Placement],
) -> float:
    """Fraction of the new item's base area resting on supporting surfaces.

    A surface is "supporting" if it is either:
      - the container floor (y == 0), or
      - the top face of an earlier placement whose ``y_max_mm`` equals the new item's
        ``y_mm`` and whose footprint overlaps the new item.
    """
    if placement.position.y_mm == 0:
        return 1.0
    base_area = placement.rotated_dimensions.base_area_mm2
    if base_area <= 0:
        return 0.0
    supported = 0
    for prev in prior_placements:
        if prev.y_max_mm != placement.position.y_mm:
            continue
        ox = max(0, min(placement.x_max_mm, prev.x_max_mm) - max(placement.position.x_mm, prev.position.x_mm))
        oz = max(0, min(placement.z_max_mm, prev.z_max_mm) - max(placement.position.z_mm, prev.position.z_mm))
        supported += ox * oz
    return supported / base_area


def _is_lifo_violation_added(
    new_placement: Placement,
    new_item: CargoItem,
    prior_placements: list[Placement],
    items_by_id: dict[str, CargoItem],
) -> bool:
    """A LIFO violation is created if the new item has a smaller ``delivery_stop``
    than an item already placed *behind it* in the container (smaller x_mm) — that
    later-loaded item blocks access to the earlier-loaded one.
    """
    stop = new_item.delivery_stop
    if stop == 0:
        return False
    for prev in prior_placements:
        prev_stop = items_by_id[prev.item_id].delivery_stop
        if prev_stop == 0 or prev_stop <= stop:
            continue
        # 'prev' is for a later drop. If it sits behind the new item (further from
        # the door), it now blocks 'new_placement'. We approximate "blocks" as
        # x-projection overlap + y/z overlap.
        x_overlap = max(
            0,
            min(new_placement.x_max_mm, prev.x_max_mm) - max(new_placement.position.x_mm, prev.position.x_mm),
        )
        z_overlap = max(
            0,
            min(new_placement.z_max_mm, prev.z_max_mm) - max(new_placement.position.z_mm, prev.position.z_mm),
        )
        if x_overlap > 0 and z_overlap > 0 and prev.position.x_mm < new_placement.position.x_mm:
            return True
    return False


def compute_reward_vector(
    *,
    new_placement: Placement,
    new_item: CargoItem,
    container: Container,
    prior_placements: list[Placement],
    items_by_id: dict[str, CargoItem],
    cog_before: CoGTracker,
    cog_after: CoGTracker,
    door_zone_fraction: float = 0.20,
    stable_threshold: float = 0.70,
    scale: tuple[float, float, float, float, float] = DEFAULT_SCALE,
) -> RewardComponents:
    """Compute one step's 5-dimensional reward vector.

    ``cog_before`` and ``cog_after`` must be the running CoG trackers immediately
    before and after the placement was added — the caller maintains both.
    """
    # 0. utilisation gain
    util_gain = new_placement.rotated_dimensions.volume_mm3 / container.internal.volume_mm3

    # 1. access efficiency: is the centroid in the door-accessible zone?
    centroid_x = new_placement.position.x_mm + new_placement.rotated_dimensions.length_mm / 2
    door_x = door_zone_fraction * container.internal.length_mm
    access_eff_gain = 1.0 if centroid_x <= door_x else 0.0

    # 2. stability: is the placement well-supported?
    support_ratio = _support_ratio_from_placement(new_placement, prior_placements)
    stability_gain = 1.0 if support_ratio >= stable_threshold else 0.0

    # 3. CoG balance: penalise drift away from the container's geometric centre.
    # |long_dev|, |lat_dev| are both in [0, 0.5]; their sum is in [0, 1]. We want
    # high reward for low deviation, so use 1 - sum, clipped to [-1, 1].
    long_dev = abs(cog_after.longitudinal_deviation)
    lat_dev = abs(cog_after.lateral_deviation)
    cog_balance_gain = max(-1.0, min(1.0, 1.0 - (long_dev + lat_dev)))

    # 4. LIFO compliance: did this placement create a violation?
    lifo_gain = 0.0 if _is_lifo_violation_added(
        new_placement, new_item, prior_placements, items_by_id,
    ) else 1.0

    return RewardComponents(
        util_gain=float(util_gain * scale[0]),
        access_eff_gain=float(access_eff_gain * scale[1]),
        stability_gain=float(stability_gain * scale[2]),
        cog_balance_gain=float(cog_balance_gain * scale[3]),
        lifo_gain=float(lifo_gain * scale[4]),
    )


def scalarise(reward_vector: list[float], preference: list[float]) -> float:
    """Linear scalarisation r' = w · r. The preference vector sums to 1 by construction."""
    return float(sum(w * r for w, r in zip(preference, reward_vector, strict=True)))
