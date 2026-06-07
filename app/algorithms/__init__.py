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

    "ppo" is kept as a backward-compat alias for "pct" so the digital-twin dashboard
    frontend — which hard-codes algorithm="ppo" for the DRL option — keeps working
    against this service after swapping the loading-service deployment from v1 to v2.
    """
    if code in ("pct", "ppo"):
        from app.algorithms.pct.pct_agent import PCTPackingAgent
        from app.config import settings
        kwargs.setdefault("weights_path", str(settings.pct_weights_path))
        return PCTPackingAgent(**kwargs)
    if code == "morl_pct":
        from app.algorithms.pct.morl_agent import MORLPCTAgent
        from app.config import settings
        kwargs.setdefault("weights_path", str(settings.pct_weights_path))
        return MORLPCTAgent(**kwargs)
    if code == "cpsat":
        from app.algorithms.cpsat import CPSATConfig, CPSATSolver
        if kwargs:
            return CPSATSolver(CPSATConfig(**kwargs))
        return CPSATSolver()
    if code not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Unknown algorithm: {code!r}. Known: "
            f"{sorted(ALGORITHM_REGISTRY) + ['pct', 'ppo', 'morl_pct', 'cpsat']}"
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
