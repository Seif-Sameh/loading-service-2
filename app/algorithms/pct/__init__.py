"""PCT (Packing Configuration Tree) — port of Zhao, Yu & Xu, ICLR 2022.

Upstream: https://github.com/alexfrom0815/Online-3D-BPP-PCT (MIT).
Cite the paper if you use this:

    @inproceedings{zhao2022learning,
        title={Learning Efficient Online 3D Bin Packing on Packing Configuration Trees},
        author={Hang Zhao and Yang Yu and Kai Xu},
        booktitle={International Conference on Learning Representations},
        year={2022},
    }

What's in this package:
- :mod:`distributions`     — drop-in PyTorch distribution helpers (FixedCategorical etc.)
- :mod:`utils`             — init helpers + observation packing/unpacking
- :mod:`graph_encoder`     — multi-head graph attention encoder
- :mod:`attention_model`   — GAT + pointer actor network (the heart of PCT)
- :mod:`pct_model`         — actor-critic wrapper (DRL_GAT)

Coming next (separate commits):
- ``pct_env``    — gymnasium env that builds tree observations from our simulator
- ``acktr``      — KFAC + ACKTR trainer with Kaggle-resumable checkpoints
- ``pct_agent``  — :class:`PackingAlgorithm` wrapper for inference
"""
from .pct_model import DRL_GAT, PCTConfig

__all__ = ["DRL_GAT", "PCTConfig"]
