"""
Noise utilities.

Currently the noise is wired directly into quantum_policy.make_policy()
via DepolarizingChannel after each gate. This file provides the canonical
list of noise levels to sweep over and brief explanations for each.

Depolarizing noise model:
    With probability p, the qubit's state is replaced by the maximally
    mixed state (uniform random). With probability (1-p), nothing happens.
    This is a simple but standard model for gate errors on superconducting
    quantum hardware.

Realistic context for noise probabilities (per gate):
    p = 0.000   ->  ideal simulator (no real hardware achieves this)
    p = 0.001   ->  best-in-class superconducting qubits, single-qubit gates
    p = 0.005   ->  typical superconducting two-qubit gate today (~2026)
    p = 0.01    ->  noisier hardware / older devices
    p = 0.05    ->  stress test — well beyond real hardware
"""

NOISE_LEVELS = [0.0, 0.005, 0.01, 0.05]

NOISE_LABELS = {
    0.0:   "noiseless (ideal)",
    0.005: "near-current hardware",
    0.01:  "noisy hardware",
    0.05:  "stress test",
}