"""Metrics: accuracy / ECE / Brier (definitions aligned with Nimble)."""


def accuracy(preds, refs):
    if not preds:
        return float("nan")
    return sum(p == r for p, r in zip(preds, refs)) / len(preds)


def expected_calibration_error(probs, corrects, n_bins=10):
    """probs: per-sample confidence in the gold class; corrects: 0/1."""
    bins = [0.0] * n_bins
    counts = [0] * n_bins
    for p, c in zip(probs, corrects):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b] += abs(p - c)
        counts[b] += 1
    total = sum(counts)
    if total == 0:
        return float("nan")
    return sum(bins[i] for i in range(n_bins) if counts[i]) / sum(c for c in counts if c)


def brier(scores, refs):
    """scores: list[dict label->prob]; refs: list[label]. Standard multiclass Brier score."""
    if not scores:
        return float("nan")
    total = 0.0
    for dist, ref in zip(scores, refs):
        total += sum((p - (1.0 if label == ref else 0.0)) ** 2 for label, p in dist.items())
    return total / len(scores)
