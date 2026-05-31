"""
Run all experiments and save results to /results.

WARNING: This is the long one. Expected runtime on a modern laptop CPU:
    - Classical: ~2 minutes per seed (5 seeds = 10 min)
    - Quantum noiseless: ~15-30 min per seed (5 seeds = 1-2.5 hours)
    - Quantum noise sweep (3 noise levels, 3 seeds each): ~3-6 hours

Suggested flow:
    1. First run with QUICK_MODE=True (fewer episodes/seeds) to verify it works.
    2. Then run full (set QUICK_MODE=False) overnight.
"""

import os
import json
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_classical, train_quantum
from src.noise import NOISE_LEVELS
from src import evaluate as ev


# ---- Toggle this for fast smoke-tests vs. full runs ----
QUICK_MODE = False   # change to False for the final run

if QUICK_MODE:
    N_SEEDS_NOISELESS = 2
    N_SEEDS_NOISE     = 2
    N_EPISODES        = 100
    NOISE_SUBSET      = [0.0, 0.01]
else:
    N_SEEDS_NOISELESS = 5
    N_SEEDS_NOISE     = 3
    N_EPISODES        = 500
    NOISE_SUBSET      = NOISE_LEVELS


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_rewards(name: str, rewards_per_seed):
    """Save a (n_seeds, n_episodes) array as CSV."""
    arr = np.array(rewards_per_seed)
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    pd.DataFrame(arr.T,
                 columns=[f"seed_{i}" for i in range(arr.shape[0])]).to_csv(path, index_label="episode")
    print(f"  -> saved {path}")


def main():
    print("="*60)
    print(f"Running experiments (QUICK_MODE={QUICK_MODE})")
    print(f"  episodes per run: {N_EPISODES}")
    print(f"  noiseless seeds:  {N_SEEDS_NOISELESS}")
    print(f"  noise seeds:      {N_SEEDS_NOISE}")
    print(f"  noise levels:     {NOISE_SUBSET}")
    print("="*60)

    # ---- 1. Classical baseline ----
    print("\n[1/3] Training classical MLP policies...")
    classical_rewards = []
    for seed in range(N_SEEDS_NOISELESS):
        _, rewards = train_classical(seed=seed, max_episodes=N_EPISODES)
        classical_rewards.append(rewards)
    save_rewards("classical_noiseless", classical_rewards)

    # ---- 2. Quantum noiseless ----
    print("\n[2/3] Training quantum policies (noiseless)...")
    quantum_noiseless_rewards = []
    for seed in range(N_SEEDS_NOISELESS):
        _, rewards = train_quantum(seed=seed, noise_prob=0.0, max_episodes=N_EPISODES)
        quantum_noiseless_rewards.append(rewards)
    save_rewards("quantum_noiseless", quantum_noiseless_rewards)

    # ---- 3. Quantum noise sweep ----
    print("\n[3/3] Quantum noise sweep...")
    for p in NOISE_SUBSET:
        if p == 0.0:
            continue   # already done above
        print(f"\n  noise p={p}")
        rewards_this_p = []
        for seed in range(N_SEEDS_NOISE):
            _, rewards = train_quantum(seed=seed, noise_prob=p, max_episodes=N_EPISODES)
            rewards_this_p.append(rewards)
        save_rewards(f"quantum_noise_p{p}", rewards_this_p)

    # ---- 4. Summary ----
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    summary_rows = [
        ev.results_summary("classical_noiseless", classical_rewards),
        ev.results_summary("quantum_noiseless",   quantum_noiseless_rewards),
    ]
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)

    # Paired t-test
    q_finals = [ev.final_performance(r) for r in quantum_noiseless_rewards]
    c_finals = [ev.final_performance(r) for r in classical_rewards]
    t, p = ev.paired_t_test(q_finals, c_finals)
    print(f"\nPaired t-test (quantum vs classical, noiseless): t={t:.3f}, p={p:.4f}")

    with open(os.path.join(RESULTS_DIR, "ttest.json"), "w") as f:
        json.dump({"t_stat": t, "p_value": p,
                   "quantum_finals": q_finals, "classical_finals": c_finals}, f, indent=2)


if __name__ == "__main__":
    main()