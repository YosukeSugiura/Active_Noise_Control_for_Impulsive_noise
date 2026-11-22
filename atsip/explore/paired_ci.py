"""Seed-bootstrap intervals for paper comparisons."""
from collections import defaultdict

import numpy as np


def paired(left, right, metric):
    """Return the mean, 95% interval and count for completed matched pairs.

    Average class differences within each seed, then resample seed means.
    Sort by seed, class and condition so input and comparison order are immaterial.
    """
    def index(rows):
        return {
            (row["seed"], row["cls"], row["scen"]): value
            for row in rows
            if not row["div"] and (value := metric(row)) is not None
        }

    a, b = index(left), index(right)
    keys = sorted(a.keys() & b.keys())
    if not keys:
        return None

    by_seed = defaultdict(lambda: defaultdict(list))
    for seed, cls, scen in keys:
        key = seed, cls, scen
        by_seed[seed][cls].append(a[key] - b[key])
    means = np.array([
        np.mean([np.mean(differences) for differences in classes.values()])
        for classes in by_seed.values()
    ])
    draws = np.random.default_rng(20261121).choice(means, (2000, len(means))).mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    return float(means.mean()), float(lo), float(hi), len(keys)
