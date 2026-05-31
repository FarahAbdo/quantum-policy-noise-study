"""
Evaluation utilities: smoothing, summary stats, paired t-tests, compute accounting.

The methodological signal to Dr. Sykes lives here. Most QRL papers skip this part.
We deliberately make it explicit.
"""

import numpy as np
import pandas as pd
from scipy import stats


def smooth(rewards, window: int = 20):
    """Trailing moving average. Use only for plotting, not for the headline number."""
    rewards = np.asarray(rewards, dtype=float)
    if len(rewards) < window:
        return rewards
    kernel = np.ones(window) / window
    return np.convolve(rewards, kernel, mode="valid")


def final_performance(rewards, last_n: int = 100):
    """
    Headline metric: mean reward over the LAST `last_n` episodes.
    More stable than 'best episode' (which is cherry-picking).
    """
    rewards = np.asarray(rewards, dtype=float)
    return float(rewards[-last_n:].mean())


def aggregate_seeds(rewards_per_seed):
    """
    Given a list of reward-curves (one per seed), return mean and std per episode.

    rewards_per_seed: list of lists, length = n_seeds, inner length = n_episodes
    """
    arr = np.array(rewards_per_seed, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def paired_t_test(quantum_finals, classical_finals):
    """
    Paired t-test of final-performance per seed.
    Returns (t_statistic, p_value).
    A small p_value (< 0.05) means the difference is unlikely due to chance.
    """
    q = np.asarray(quantum_finals, dtype=float)
    c = np.asarray(classical_finals, dtype=float)
    t_stat, p_val = stats.ttest_rel(q, c)
    return float(t_stat), float(p_val)


def compute_accounting(circuit_specs: dict):
    """
    Build a one-row summary table of the compute cost of a circuit.

    Use qml.specs(circuit)(state, params) in your notebook to get the spec dict,
    then pass it here.
    """
    return pd.DataFrame([{
        "qubits":         circuit_specs.get("num_used_wires", "?"),
        "gates":          circuit_specs.get("num_operations", "?"),
        "depth":          circuit_specs.get("depth", "?"),
        "trainable_params": circuit_specs.get("num_trainable_params", "?"),
    }])


def results_summary(label, rewards_per_seed):
    """Compact one-line summary for the results table."""
    finals = [final_performance(r) for r in rewards_per_seed]
    return {
        "label":          label,
        "n_seeds":        len(rewards_per_seed),
        "final_mean":     float(np.mean(finals)),
        "final_std":      float(np.std(finals)),
        "final_min":      float(np.min(finals)),
        "final_max":      float(np.max(finals)),
    }