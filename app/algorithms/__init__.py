"""Packing algorithms — heuristics, GA, PCT (single-objective), MORL-PCT, and CP-SAT.

All algorithms implement :class:`PackingAlgorithm` so the solver can treat them uniformly.
"""
from .base import PackingAlgorithm, solve
from .ga import GeneticAlgorithm
from .heuristics import (
    BestAreaFit,
    BestLongestSideFit,
    BestShortestSideFit,
    BottomLeft,
    ExtremePoints,
)

ALGORITHM_REGISTRY: dict[str, type[PackingAlgorithm]] = {
    "baf": BestAreaFit,
    "bssf": BestShortestSideFit,
    "blsf": BestLongestSideFit,
    "bl": BottomLeft,
    "extreme_points": ExtremePoints,
    "ga": GeneticAlgorithm,
}


def get_algorithm(code: str, **kwargs) -> PackingAlgorithm:
    """Instantiate an algorithm by registry code.

    "pct", "morl_pct", and "cpsat" are loaded lazily so this function works in environments
    without PyTorch / OR-Tools installed.
    """
    if code == "pct":
        from app.algorithms.pct.pct_agent import PCTPackingAgent
        return PCTPackingAgent(**kwargs)
    if code == "morl_pct":
        from app.algorithms.pct.morl_agent import MORLPCTAgent
        return MORLPCTAgent(**kwargs)
    if code == "cpsat":
        from app.algorithms.cpsat import CPSATConfig, CPSATSolver
        if kwargs:
            return CPSATSolver(CPSATConfig(**kwargs))
        return CPSATSolver()
    if code not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Unknown algorithm: {code!r}. Known: "
            f"{sorted(ALGORITHM_REGISTRY) + ['pct', 'morl_pct', 'cpsat']}"
        )
    return ALGORITHM_REGISTRY[code]()


__all__ = [
    "ALGORITHM_REGISTRY",
    "BestAreaFit",
    "BestLongestSideFit",
    "BestShortestSideFit",
    "BottomLeft",
    "ExtremePoints",
    "GeneticAlgorithm",
    "PackingAlgorithm",
    "get_algorithm",
    "solve",
]
