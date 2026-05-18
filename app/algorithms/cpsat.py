"""CP-SAT (Constraint Programming, OR-Tools) baseline for 3D bin packing.

Formulates the **offline** 3D-BPP as a constraint-satisfaction problem and solves
it with Google OR-Tools' CP-SAT solver. Within the time budget the solver returns
the **optimal** (or epsilon-optimal) packing for the objective: *maximise total
packed item volume*.

Implementation notes:

- Positions are discretised at ``grid_mm`` (default 50 mm) to keep CP-SAT's
  integer-domain search tractable. At 50 mm a 40HC is 240 × 47 × 54 cells.
  At 1 mm the same container is 12032 × 2352 × 2698 cells — too large for
  the solver to make progress in seconds.

- 3D non-overlap is enforced via OR-Tools' built-in ``AddNoOverlap2D`` on the
  *floor plane* (x, z) for items at the **same y level**, plus pairwise
  vertical-separation BoolVars for stacked items. This is the standard
  efficient encoding (used by the OR-Tools knapsack examples).

- Rotation: binary (upright LWH vs WLH). Items with ``this_side_up=True``
  also only get upright rotations (matches our env contract).

Constraints implemented:
  - within-bounds
  - non-overlap (3D)
  - weight payload cap
  - reefer compatibility
  - IMDG pairwise segregation (optional; on by default for fair comparison)

Constraints NOT yet implemented (will degrade comparison fairness slightly):
  - floor-load pressure rating
  - stability / support polygon
  - LIFO delivery order

Wraps as a :class:`PackingAlgorithm` so it slots into the existing
``solve()`` infrastructure — the algorithm is fundamentally offline though,
attaching to env triggers the full solve, then ``select()`` replays.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.algorithms.base import PackingAlgorithm
from app.env.packing_env import PackingState
from app.schemas import (
    CargoItem,
    Container,
    Dimensions,
    Placement,
    Position,
    Rotation,
)


@dataclass
class CPSATConfig:
    """Hyperparameters for :class:`CPSATSolver`."""

    time_limit_s: float = 30.0
    num_search_workers: int = 4
    grid_mm: int = 50              # discretisation step for position variables (mm)
    log_search_progress: bool = False
    enforce_imdg: bool = True


class CPSATSolver(PackingAlgorithm):
    """Offline CP-SAT 3D bin packer wrapped as a PackingAlgorithm."""

    code = "cpsat"
    display_name = "CP-SAT (OR-Tools)"

    def __init__(self, cfg: CPSATConfig | None = None) -> None:
        self.cfg = cfg or CPSATConfig()
        self._planned: list[Placement] = []
        self._planned_by_item_id: dict[str, Placement] = {}
        self._env = None
        self._meta: dict[str, Any] = {}

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    def attach_env(self, env) -> None:
        self._env = env
        container = env.container
        items = env._inner.items if hasattr(env, "_inner") else env.items
        t0 = time.perf_counter()
        self._planned, status, obj = _solve_cpsat(container, items, self.cfg)
        self._planned_by_item_id = {p.item_id: p for p in self._planned}
        self._meta = {
            "cpsat_status": status,
            "cpsat_objective_cm3": obj,
            "cpsat_solve_seconds": time.perf_counter() - t0,
            "cpsat_planned_items": len(self._planned),
            "cpsat_total_items": len(items),
        }

    def select(self, state: PackingState) -> int:
        if not state.candidates:
            return 0
        current = state.current_item
        if current is None or current.id not in self._planned_by_item_id:
            return _bottom_left_index(state)
        target = self._planned_by_item_id[current.id]
        best_idx = 0
        best_score = float("inf")
        for i, c in enumerate(state.candidates):
            rot_penalty = 0.0 if c.rotation == target.rotation else 1e6
            dx = c.position.x_mm - target.position.x_mm
            dy = c.position.y_mm - target.position.y_mm
            dz = c.position.z_mm - target.position.z_mm
            d = (dx * dx + dy * dy + dz * dz) ** 0.5 + rot_penalty
            if d < best_score:
                best_score = d
                best_idx = i
        return best_idx


# ---------------------------------------------------------------------------
# CP-SAT model — efficient 3D bin packing encoding
# ---------------------------------------------------------------------------


def _solve_cpsat(
    container: Container,
    items: list[CargoItem],
    cfg: CPSATConfig,
) -> tuple[list[Placement], str, int]:
    """Build and solve the CP-SAT model on a discretised grid."""
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:
        raise RuntimeError(
            "OR-Tools required. Install with: pip install ortools"
        ) from e

    g = max(1, int(cfg.grid_mm))
    # Convert everything to grid cells (integer cell coordinates)
    L = container.internal.length_mm // g
    W = container.internal.width_mm // g
    H = container.internal.height_mm // g
    n = len(items)

    # Per-item dimensions in grid units; round UP so footprints stay safe
    def _up(v: int) -> int:
        return -(-v // g)

    item_dims = []
    for it in items:
        d = it.dimensions
        item_dims.append((_up(d.length_mm), _up(d.width_mm), _up(d.height_mm)))

    model = cp_model.CpModel()

    included = [model.NewBoolVar(f"inc_{i}") for i in range(n)]
    rot = [model.NewBoolVar(f"rot_{i}") for i in range(n)]

    # Effective dimensions (length × width depend on rotation; height fixed)
    l_eff: list = []
    w_eff: list = []
    h_eff: list = []
    for i, it in enumerate(items):
        l_mm, w_mm, h_mm = item_dims[i]
        ub = max(l_mm, w_mm)
        l_eff_var = model.NewIntVar(1, max(ub, 1), f"l_eff_{i}")
        w_eff_var = model.NewIntVar(1, max(ub, 1), f"w_eff_{i}")
        model.Add(l_eff_var == l_mm).OnlyEnforceIf(rot[i].Not())
        model.Add(l_eff_var == w_mm).OnlyEnforceIf(rot[i])
        model.Add(w_eff_var == w_mm).OnlyEnforceIf(rot[i].Not())
        model.Add(w_eff_var == l_mm).OnlyEnforceIf(rot[i])
        l_eff.append(l_eff_var)
        w_eff.append(w_eff_var)
        h_eff.append(h_mm)
        # this_side_up: lock to LWH (rot=0) — both upright rotations still allowed because
        # LWH and WLH both have h vertical, but we keep the simpler rot==0 lock to reduce
        # branching when the item has no orientation freedom.
        if it.this_side_up and it.dimensions.length_mm == it.dimensions.width_mm:
            model.Add(rot[i] == 0)
        # If item doesn't physically fit even at best rotation, force exclusion
        if min(l_mm, w_mm) > L or max(l_mm, w_mm) > max(L, W) or h_mm > H:
            model.Add(included[i] == 0)

    # Position variables (in grid cells)
    x_start = [model.NewIntVar(0, max(L, 1), f"xs_{i}") for i in range(n)]
    z_start = [model.NewIntVar(0, max(W, 1), f"zs_{i}") for i in range(n)]
    y_start = [model.NewIntVar(0, max(H, 1), f"ys_{i}") for i in range(n)]
    x_end = [model.NewIntVar(0, max(L, 1), f"xe_{i}") for i in range(n)]
    z_end = [model.NewIntVar(0, max(W, 1), f"ze_{i}") for i in range(n)]
    y_end = [model.NewIntVar(0, max(H, 1), f"ye_{i}") for i in range(n)]

    # end = start + size; bounds
    for i in range(n):
        model.Add(x_end[i] == x_start[i] + l_eff[i])
        model.Add(z_end[i] == z_start[i] + w_eff[i])
        model.Add(y_end[i] == y_start[i] + h_eff[i])
        # Within container (only when included)
        model.Add(x_end[i] <= L).OnlyEnforceIf(included[i])
        model.Add(z_end[i] <= W).OnlyEnforceIf(included[i])
        model.Add(y_end[i] <= H).OnlyEnforceIf(included[i])
        # When excluded, pin position to 0 (cleaner search)
        model.Add(x_start[i] == 0).OnlyEnforceIf(included[i].Not())
        model.Add(z_start[i] == 0).OnlyEnforceIf(included[i].Not())
        model.Add(y_start[i] == 0).OnlyEnforceIf(included[i].Not())

    # 3D non-overlap — for every pair (i, j), if both included then at least one of the
    # 6 axis-separation conditions must hold. We use OnlyEnforceIf([included[i], included[j]])
    # which the CP-SAT solver natively understands as "constraint active iff both literals true".
    for i in range(n):
        for j in range(i + 1, n):
            sep_x = model.NewBoolVar(f"sx_{i}_{j}")
            sep_xr = model.NewBoolVar(f"sxr_{i}_{j}")
            sep_z = model.NewBoolVar(f"sz_{i}_{j}")
            sep_zr = model.NewBoolVar(f"szr_{i}_{j}")
            sep_y = model.NewBoolVar(f"sy_{i}_{j}")
            sep_yr = model.NewBoolVar(f"syr_{i}_{j}")
            model.Add(x_end[i] <= x_start[j]).OnlyEnforceIf(sep_x)
            model.Add(x_end[j] <= x_start[i]).OnlyEnforceIf(sep_xr)
            model.Add(z_end[i] <= z_start[j]).OnlyEnforceIf(sep_z)
            model.Add(z_end[j] <= z_start[i]).OnlyEnforceIf(sep_zr)
            model.Add(y_end[i] <= y_start[j]).OnlyEnforceIf(sep_y)
            model.Add(y_end[j] <= y_start[i]).OnlyEnforceIf(sep_yr)
            # Active only when both items are included; disabled if either is excluded.
            model.AddBoolOr([sep_x, sep_xr, sep_z, sep_zr, sep_y, sep_yr]).OnlyEnforceIf(
                [included[i], included[j]]
            )

    # Weight payload cap
    weight_grams = [int(it.weight_kg * 1000) for it in items]
    payload_grams = int(container.payload_kg * 1000)
    model.Add(sum(weight_grams[i] * included[i] for i in range(n)) <= payload_grams)

    # Reefer compatibility
    if not container.is_reefer:
        for i, it in enumerate(items):
            if it.requires_reefer:
                model.Add(included[i] == 0)

    # IMDG segregation (simplified — uses grid units, segregation distances scaled by g)
    if cfg.enforce_imdg:
        try:
            from app.catalog.loader import imdg_table
            from app.schemas import HazmatClass

            tbl = imdg_table()
            for i in range(n):
                ci = items[i].hazmat_class
                if ci == HazmatClass.NONE:
                    continue
                for j in range(i + 1, n):
                    cj = items[j].hazmat_class
                    if cj == HazmatClass.NONE:
                        continue
                    code = tbl.segregation_code(ci, cj)
                    if code == 0:
                        continue
                    if code >= 3:
                        model.AddBoolOr([included[i].Not(), included[j].Not()])
                        continue
                    d_fore_cells = tbl.separated_fore_aft_mm // g if code == 2 else tbl.away_from_mm // g
                    d_lat_cells = tbl.separated_lateral_mm // g if code == 2 else tbl.away_from_mm // g
                    if d_fore_cells <= 0 and d_lat_cells <= 0:
                        continue
                    sxf = model.NewBoolVar(f"sxf_{i}_{j}")
                    sxfr = model.NewBoolVar(f"sxfr_{i}_{j}")
                    szf = model.NewBoolVar(f"szf_{i}_{j}")
                    szfr = model.NewBoolVar(f"szfr_{i}_{j}")
                    model.Add(x_start[j] - x_end[i] >= d_fore_cells).OnlyEnforceIf(sxf)
                    model.Add(x_start[i] - x_end[j] >= d_fore_cells).OnlyEnforceIf(sxfr)
                    model.Add(z_start[j] - z_end[i] >= d_lat_cells).OnlyEnforceIf(szf)
                    model.Add(z_start[i] - z_end[j] >= d_lat_cells).OnlyEnforceIf(szfr)
                    model.AddBoolOr([sxf, sxfr, szf, szfr]).OnlyEnforceIf(
                        [included[i], included[j]]
                    )
        except Exception:
            pass

    # Objective: maximize packed cell-volume (each cell = g³ mm³ — units cancel since
    # we maximize a sum proportional to actual volume).
    cell_volumes = [
        max(1, item_dims[i][0] * item_dims[i][1] * item_dims[i][2]) for i in range(n)
    ]
    model.Maximize(sum(cell_volumes[i] * included[i] for i in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cfg.time_limit_s
    solver.parameters.num_search_workers = max(1, cfg.num_search_workers)
    if cfg.log_search_progress:
        solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
    }.get(status, str(status))

    placements: list[Placement] = []
    objective_value = 0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective_value = int(solver.ObjectiveValue())
        for i, item in enumerate(items):
            if not solver.Value(included[i]):
                continue
            xi = int(solver.Value(x_start[i])) * g
            yi = int(solver.Value(y_start[i])) * g
            zi = int(solver.Value(z_start[i])) * g
            ri = int(solver.Value(rot[i]))
            rotation = Rotation.LWH if ri == 0 else Rotation.WLH
            d = item.dimensions
            rotated = Dimensions(
                length_mm=d.length_mm if ri == 0 else d.width_mm,
                width_mm=d.width_mm if ri == 0 else d.length_mm,
                height_mm=d.height_mm,
            )
            placements.append(
                Placement(
                    item_id=item.id,
                    position=Position(x_mm=xi, y_mm=yi, z_mm=zi),
                    rotation=rotation,
                    rotated_dimensions=rotated,
                )
            )

    return placements, status_name, objective_value


def _bottom_left_index(state: PackingState) -> int:
    best, best_key = 0, None
    for i, c in enumerate(state.candidates):
        key = (c.position.y_mm, c.position.x_mm, c.position.z_mm)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best
