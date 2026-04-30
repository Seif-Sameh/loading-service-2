# loading-service-2

Container-loading optimization microservice for the Alexandria Port Digital Twin — **v2 rebuild** around **PCT (Packing Configuration Tree)**, the ICLR 2022 method that achieves 76 % space utilisation on standard 3D-BPP benchmarks vs ~60 % for the strongest classical heuristic (Deep-Bottom-Left).

> **Reference**: Zhao, Yu & Xu — *Learning Efficient Online 3D Bin Packing on Packing Configuration Trees*, ICLR 2022. Code: https://github.com/alexfrom0815/Online-3D-BPP-PCT.

## Why a v2 rebuild

`loading-service` (v1) used a PPO + Transformer policy trained on corner-point candidates. It plateaued at heuristic-parity (~25 % vs best heuristic's ~26 % on Alexandria-realistic voyages, ~71 % vs 77 % on the BR academic benchmark) because the state representation, action head, and RL algorithm together left no room above the heuristic baseline.

PCT fixes this with three structural changes:

| Component | v1 | v2 (PCT) |
|---|---|---|
| State | Heightmap + corner-point list | **Tree of placed items + tree of leaf candidates** |
| Encoder | Self-attention over EMSs | **Graph Attention Network (GAT)** over the tree |
| Action head | Linear over fixed slots | **Pointer network** over variable leaf nodes |
| Candidate generator | Cross-product corner points | **Event Points** (start/end of every placed box × each axis) |
| RL algorithm | PPO | **ACKTR** (Actor-Critic with Kronecker-factored Trust Region) |

In their ICLR paper, on a 10×10×10 discrete bin with stability:

| Method | Utilisation |
|---|---:|
| Random | 36.7 % |
| Best heuristic (Deep-Bottom-Left) | 60.5 % |
| Prior DRL (Zhao 2021, AAAI) | 70.9 % |
| **PCT + Event Points** | **76.0 %** (+15.5 pp over heuristic) |

## What carried over from v1

| Layer | Status |
|---|---|
| Pydantic schemas | Kept verbatim |
| Container catalog (8 ISO containers + IMDG matrix + 13 cargo presets) | Kept verbatim |
| Constraint layer (weight, payload, floor-load, IMDG, reefer, stability, CoG) | Kept verbatim |
| Data layer (BR1-10, Wadaboa pool, Alexandria sampler) | Kept verbatim |
| Heuristic baselines (BAF, BSSF, BLSF, Bottom-Left, Extreme Points) | Kept verbatim |
| GA baseline | Kept verbatim |
| FastAPI service + WebSocket streaming | Kept verbatim |
| Heuristic env + heightmap | Kept verbatim (used by heuristics + GA) |
| Tests (44 passing) | Kept verbatim |

## What's new in v2

Will land in `app/algorithms/pct/`:
- `graph_encoder.py` — GAT layers (ported from PCT repo)
- `attention_model.py` — pointer attention with skip-connections (ported)
- `pct_model.py` — actor-critic wrapper
- `kfac.py` + `acktr.py` — ACKTR optimiser + trainer
- `pct_env.py` — gymnasium env producing PCT tree observations using Event Points
- `pct_agent.py` — `PackingAlgorithm` wrapper for inference

## Project layout

```
loading-service-2/
├── app/
│   ├── schemas.py                # Pydantic DTOs (carry-over)
│   ├── catalog/                  # ISO containers + cargo + IMDG (carry-over)
│   ├── constraints/              # mask + reward + CoG + IMDG (carry-over)
│   ├── data/                     # BR + Wadaboa + Alexandria sampler (carry-over)
│   ├── env/                      # heightmap + heuristic env (carry-over)
│   ├── algorithms/
│   │   ├── heuristics.py         # 5 baselines (carry-over)
│   │   ├── ga.py                 # GA baseline (carry-over)
│   │   └── pct/                  # NEW — PCT model + ACKTR trainer + env adapter
│   ├── services/, api/, main.py  # FastAPI (carry-over)
│   └── utils/
├── data/                         # Brunel BR + Wadaboa parquet (carry-over)
├── notebooks/                    # NEW — PCT training + eval notebooks
├── scripts/                      # prepare_datasets.py + future train script (carry-over)
├── tests/                        # 44 passing (carry-over) + future PCT tests
└── models/                       # trained checkpoints land here
```

## Quickstart (dev)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install-dev
python -m scripts.prepare_datasets --wadaboa-pkl <path/to/products.pkl>
make test          # 44 tests, all green
make run           # FastAPI on :8009
```

## Training the PCT policy

The training notebook (`notebooks/01_train_pct.ipynb`, coming next) is built around your **Kaggle 30 h GPU budget** with multi-session checkpointing:

- Autosave every N iterations to `/kaggle/working/` so a session timeout never loses progress
- `RESUME_FROM_CHECKPOINT = True` automatically picks up where the last session left off
- Total target: ~10 M ACKTR steps split across 3-4 Kaggle sessions of ~8 h each

PCT's paper reports ~1 day on a single TITAN V; on a Kaggle T4 expect 2-3 sessions to converge.

## License

MIT (this repo).
PCT code adapted from https://github.com/alexfrom0815/Online-3D-BPP-PCT — also MIT.
