"""Patched torch distributions used by the PCT actor.

Copied verbatim (only renamed for namespace) from
https://github.com/alexfrom0815/Online-3D-BPP-PCT/blob/master/distributions.py.
"""
from __future__ import annotations

import torch

# Categorical with a few helper methods used in the PCT actor head.
FixedCategorical = torch.distributions.Categorical

_old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: _old_sample(self).unsqueeze(-1)

_old_log_prob = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: _old_log_prob(
    self, actions.squeeze(-1)
).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)
