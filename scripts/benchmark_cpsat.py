"""CP-SAT vs heuristics vs (optionally) MORL-PCT benchmark.

Runs a fixed voyage suite across:
  - 5 classical heuristics (BAF, BSSF, BLSF, Bottom-Left, Extreme Points)
  - Genetic Algorithm
  - CP-SAT (offline exact solver)
  - (optional) MORL-PCT, when a checkpoint path is provided

Reports per-voyage and aggregate (mean / std) metrics:
  - utilisation %
  - placed %
  - access efficiency
  - stability score
  - CoG centring
  - wall-clock seconds

Usage (from the loading-service-2 root):
    python -m scripts.benchmark_cpsat                # heuristics + GA + CP-SAT
    python -m scripts.benchmark_cpsat --morl PATH    # also include MORL agent
    python -m scripts.benchmark_cpsat --items 30     # smaller voyages
    python -m scripts.benchmark_cpsat --voyages 10   # fewer voyages
    python -m scripts.benchmark_cpsat --cpsat-time 60  # CP-SAT budget per voyage

Outputs a markdown table on stdout and writes the per-voyage CSV to
``benchmarks/out/cpsat_benchmark_<timestamp>.csv``.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.algorithms import get_algorithm
from app.algorithms.base import solve
from app.catalog.loader import get_container
from app.data.alexandria_sampler import AlexandriaSampler, SamplerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "benchmarks" / "out"


@dataclass
class Result:
    voyage: int
    algorithm: str
    util_pct: float
    placed_pct: float
    access_eff: float
    stability_score: float
    cog_long_abs: float
    weight_pct: float
    elapsed_s: float
    cpsat_status: str = ""


def _access_eff(placements, container) -> float:
    if not placements:
        return 0.0
    L = container.internal.length_mm
    door = 0.20 * L
    return sum(
        1
        for p in placements
        if (p.position.x_mm + p.rotated_dimensions.length_mm / 2) <= door
    ) / len(placements)


def _stability(placed_count: int, unstable: int) -> float:
    if placed_count == 0:
        return 0.0
    return max(0.0, (placed_count - unstable) / placed_count)


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building voyage suite: {args.voyages} × {args.items} items, container={args.container}")
    sampler = AlexandriaSampler(SamplerConfig(n_items=args.items, strategy="mixed", seed=args.seed))
    cont = get_container(args.container)
    voyages = [(cont, sampler.sample()) for _ in range(args.voyages)]

    # Algorithms list
    algo_codes: list[tuple[str, dict]] = [
        ("bl", {}),
        ("extreme_points", {}),
        ("baf", {}),
        ("bssf", {}),
        ("blsf", {}),
        ("ga", {}),
        ("cpsat", {"time_limit_s": float(args.cpsat_time), "num_search_workers": int(args.cpsat_workers),
                   "enforce_imdg": True}),
    ]
    if args.morl:
        algo_codes.append(("morl_pct", {
            "weights_path": args.morl,
            "preference": [0.7, 0.1, 0.1, 0.05, 0.05],  # util-leaning balanced
        }))

    results: list[Result] = []
    for vi, (c, items) in enumerate(voyages):
        items_by_id = {it.id: it for it in items}
        print(f"\n=== voyage {vi+1}/{len(voyages)}  ({len(items)} items) ===")
        for code, kwargs in algo_codes:
            algo = get_algorithm(code, **kwargs)
            t0 = time.perf_counter()
            if hasattr(algo, "prepare"):
                algo.prepare(c, items)
            res, _ = solve(algorithm=algo, container=c, items=items)
            elapsed = time.perf_counter() - t0
            k = res.kpis
            ae = _access_eff(res.placements, c)
            ss = _stability(len(res.placements), k.unstable_count)
            cpsat_status = ""
            if code == "cpsat":
                cpsat_status = str(algo.meta.get("cpsat_status", ""))
            r = Result(
                voyage=vi,
                algorithm=code,
                util_pct=100 * k.utilization,
                placed_pct=100 * len(res.placements) / max(len(items), 1),
                access_eff=ae,
                stability_score=ss,
                cog_long_abs=abs(k.cog_long_dev),
                weight_pct=100 * k.weight_used,
                elapsed_s=elapsed,
                cpsat_status=cpsat_status,
            )
            results.append(r)
            tag = f" [{cpsat_status}]" if cpsat_status else ""
            print(
                f"  {code:<16} util {r.util_pct:>6.2f}%  placed {r.placed_pct:>6.2f}%  "
                f"AE {r.access_eff:>5.2f}  SS {r.stability_score:>5.2f}  "
                f"t {r.elapsed_s:>6.2f}s{tag}"
            )

    # Aggregate
    print("\n\n=== AGGREGATE (mean across voyages) ===")
    print(
        f'{"algorithm":<16} {"util%":>7} {"std":>5} {"placed%":>8} {"AE":>5} {"SS":>5} '
        f'{"|CoG|":>6} {"wt%":>5} {"s":>6}'
    )
    print("-" * 75)
    agg = {}
    for r in results:
        agg.setdefault(r.algorithm, []).append(r)
    for code, _ in algo_codes:
        rs = agg.get(code, [])
        if not rs:
            continue
        utils = [x.util_pct for x in rs]
        u_mean = statistics.fmean(utils)
        u_std = statistics.stdev(utils) if len(utils) > 1 else 0.0
        print(
            f"{code:<16} {u_mean:>7.2f} {u_std:>5.2f} "
            f"{statistics.fmean(x.placed_pct for x in rs):>8.2f} "
            f"{statistics.fmean(x.access_eff for x in rs):>5.2f} "
            f"{statistics.fmean(x.stability_score for x in rs):>5.2f} "
            f"{statistics.fmean(x.cog_long_abs for x in rs):>6.3f} "
            f"{statistics.fmean(x.weight_pct for x in rs):>5.2f} "
            f"{statistics.fmean(x.elapsed_s for x in rs):>6.2f}"
        )

    # CSV
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"cpsat_benchmark_{stamp}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "voyage", "algorithm", "util_pct", "placed_pct", "access_eff",
            "stability_score", "cog_long_abs", "weight_pct", "elapsed_s", "cpsat_status",
        ])
        for r in results:
            w.writerow([
                r.voyage, r.algorithm, f"{r.util_pct:.4f}", f"{r.placed_pct:.4f}",
                f"{r.access_eff:.4f}", f"{r.stability_score:.4f}",
                f"{r.cog_long_abs:.4f}", f"{r.weight_pct:.4f}", f"{r.elapsed_s:.3f}",
                r.cpsat_status,
            ])
    print(f"\n→ wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voyages", type=int, default=5)
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--container", default="40HC")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cpsat-time", type=float, default=30.0, help="CP-SAT time limit per voyage (s)")
    ap.add_argument("--cpsat-workers", type=int, default=4)
    ap.add_argument("--morl", type=str, default=None,
                    help="optional: path to morl_pct_latest.pt to include the MORL agent")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
