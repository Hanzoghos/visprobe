"""Temperature calibration (mirrors nimble/scoring/calibration.py): fit the softmax
temperature to minimize NLL.

Status: code present, but NOT exercised in the released benchmark (no reported
results / no fitted temperature shipped). Planned as future work.
"""

import torch


def fit_temperature(logits, gold_indices, ts=None):
    """logits: list[list[float]] candidate logits; gold_indices: list[int] gold indices.

    Returns (best_T, best_nll). After fitting on the calibration set, override
    config.TEMPERATURE with best_T.
    """
    ts = ts or [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    logits_t = torch.tensor(logits, dtype=torch.float32)
    gold = torch.tensor(gold_indices, dtype=torch.long)
    best_t, best_nll = 1.0, float("inf")
    for t in ts:
        logp = torch.log_softmax(logits_t / t, dim=-1)
        nll = -logp[torch.arange(len(gold)), gold].mean().item()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t, best_nll
