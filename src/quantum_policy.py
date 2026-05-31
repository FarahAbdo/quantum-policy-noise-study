"""
Variational Quantum Policy (VQP) for CartPole.

The policy is a parametrized quantum circuit:
    state (4 numbers) --> [encoding] --> [trainable layers] --> [measure] --> action probabilities

Architecture:
    - 4 qubits (one per CartPole state dimension)
    - Angle encoding: each state value rotates one qubit
    - 2 layers of hardware-efficient ansatz (RY + RZ + CNOT ring)
    - Measure expectation value of Z on qubits 0 and 1
    - Softmax those two values -> probability of action 0 vs action 1

Parameter count: 4 qubits * 2 rotations (RY, RZ) * 2 layers = 16
This matches the classical MLP baseline exactly.
"""

import pennylane as qml
import numpy as np

# ---- Hyperparameters (kept here so they're easy to find) ----
N_QUBITS = 4          # one qubit per CartPole state dim
N_LAYERS = 2          # depth of the variational ansatz
N_PARAMS = N_QUBITS * 2 * N_LAYERS   # = 16

# Action space size (CartPole has 2 actions: push left, push right)
N_ACTIONS = 2


def make_policy(noise_prob: float = 0.0):
    """
    Build the quantum policy as a PennyLane QNode.

    Args:
        noise_prob: probability of depolarizing noise after each gate.
                    0.0 = noiseless ideal simulator.
                    >0  = simulates an imperfect quantum computer.

    Returns:
        A function: policy(state, params) -> [logit_action_0, logit_action_1]
    """
    # If noise > 0 we need a 'mixed state' simulator (slower but supports noise).
    # If noise == 0 we use the fast pure-state simulator.
    device_name = "default.mixed" if noise_prob > 0 else "default.qubit"
    dev = qml.device(device_name, wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    def circuit(state, params):
        # ---- STEP 1: ENCODING ----
        # Map each of the 4 state values onto a qubit rotation.
        # qml.RY(angle, wire) rotates that qubit by `angle` around the Y axis.
        # We use arctan to squash unbounded CartPole values into [-pi/2, pi/2]
        # so encoding stays well-defined for any state.
        for i in range(N_QUBITS):
            qml.RY(np.arctan(state[i]), wires=i)
            if noise_prob > 0:
                qml.DepolarizingChannel(noise_prob, wires=i)

        # ---- STEP 2: TRAINABLE LAYERS (the "ansatz") ----
        # params has shape (N_LAYERS, N_QUBITS, 2): two rotation angles per qubit per layer.
        for layer in range(N_LAYERS):
            # 2a: per-qubit rotations (RY then RZ). These are the trainable parameters.
            for q in range(N_QUBITS):
                qml.RY(params[layer, q, 0], wires=q)
                qml.RZ(params[layer, q, 1], wires=q)
                if noise_prob > 0:
                    qml.DepolarizingChannel(noise_prob, wires=q)

            # 2b: entanglement — a "ring" of CNOTs (0->1, 1->2, 2->3, 3->0).
            # Entanglement is what makes the quantum circuit non-trivially "quantum".
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
                if noise_prob > 0:
                    qml.DepolarizingChannel(noise_prob, wires=(q + 1) % N_QUBITS)

        # ---- STEP 3: MEASUREMENT ----
        # We read out the Pauli-Z expectation value on two qubits.
        # Each returns a number in [-1, +1]. We'll softmax these to get action probs.
        return [qml.expval(qml.PauliZ(i)) for i in range(N_ACTIONS)]

    return circuit


def softmax(x):
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def action_probs(circuit, state, params):
    """
    Run the circuit and convert raw expectation values into action probabilities.
    """
    logits = circuit(state, params)
    # PennyLane returns a list/array; convert to np array
    logits = np.array([float(l) for l in logits])
    return softmax(logits)


def init_params(seed: int = 0):
    """
    Initialize the 16 trainable parameters.
    Small random values near zero — this is the standard NISQ-friendly init
    and helps avoid barren plateaus on small circuits.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.1, size=(N_LAYERS, N_QUBITS, 2))


def count_parameters():
    """Return the total trainable parameter count for the report."""
    return N_PARAMS