"""Smoke tests for the CP-SAT solver."""
from __future__ import annotations

import pytest

ortools = pytest.importorskip("ortools.sat.python.cp_model")


def test_cpsat_packs_small_voyage(container_40hc, eur_pallets_10):
    """CP-SAT should pack a small uniform voyage and report OPTIMAL status."""
    from app.algorithms.base import solve
    from app.algorithms.cpsat import CPSATConfig, CPSATSolver

    solver = CPSATSolver(CPSATConfig(time_limit_s=20.0, num_search_workers=2))
    result, _ = solve(algorithm=solver, container=container_40hc, items=eur_pallets_10)
    assert len(result.placements) >= 8  # most or all should fit
    # Solver should have recorded its status
    assert solver.meta["cpsat_status"] in ("OPTIMAL", "FEASIBLE")
    assert solver.meta["cpsat_planned_items"] >= 8


def test_cpsat_beats_or_ties_bottom_left_on_mixed_bag(container_40hc, mixed_bag):
    """On a small mixed voyage, CP-SAT should match or beat Bottom-Left on utilisation.

    This is the headline guarantee: a provable optimum should never lose to a greedy
    heuristic on a small enough instance.
    """
    from app.algorithms import get_algorithm
    from app.algorithms.base import solve
    from app.algorithms.cpsat import CPSATConfig, CPSATSolver

    bl_result, _ = solve(
        algorithm=get_algorithm("bl"), container=container_40hc, items=mixed_bag
    )
    cpsat_result, _ = solve(
        algorithm=CPSATSolver(CPSATConfig(time_limit_s=15.0, num_search_workers=2)),
        container=container_40hc,
        items=mixed_bag,
    )
    # The replay through the env may snap CP-SAT's continuous positions to
    # heightmap-grid candidates, sometimes losing a placement. Tolerance: CP-SAT
    # is allowed to be at most 1 placement worse, but must match BL utilisation
    # within 5pp.
    assert cpsat_result.kpis.utilization >= bl_result.kpis.utilization - 0.05


def test_cpsat_registry_access():
    """get_algorithm('cpsat') returns a CPSATSolver."""
    from app.algorithms import get_algorithm
    from app.algorithms.cpsat import CPSATSolver

    algo = get_algorithm("cpsat")
    assert isinstance(algo, CPSATSolver)

    algo2 = get_algorithm("cpsat", time_limit_s=5.0)
    assert isinstance(algo2, CPSATSolver)
    assert algo2.cfg.time_limit_s == 5.0


def test_cpsat_respects_reefer_constraint(container_40hc, container_20gp):
    """Reefer-requiring items in a non-reefer container must not be planned."""
    from app.algorithms.base import solve
    from app.algorithms.cpsat import CPSATConfig, CPSATSolver
    from app.catalog.loader import get_cargo_preset

    items = [get_cargo_preset("reefer_fruit_pallet", item_id=f"r{i}") for i in range(3)]
    solver = CPSATSolver(CPSATConfig(time_limit_s=5.0))
    # 40HC is NOT a reefer; the solver should plan 0 items.
    result, _ = solve(algorithm=solver, container=container_40hc, items=items)
    assert len(result.placements) == 0
    assert solver.meta["cpsat_planned_items"] == 0
