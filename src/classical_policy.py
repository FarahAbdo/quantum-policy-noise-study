"""
Classical MLP policy for CartPole — the matched-parameter-budget baseline.

We deliberately match the quantum policy's parameter count exactly (16 params),
so any performance difference is NOT explained by classical having more capacity.

Architecture: 4 -> 2 -> 2 (with no biases on the hidden layer to hit 16 params exactly)
    Layer 1: 4 * 2 = 8 weights
    Layer 2: 2 * 2 = 4 weights, + 2 + 2 = 4 biases  -> 8 weights + 4 biases = 12
    Wait — let's count again with biases:

    W1: 4 -> 2  ->  4*2 = 8 weights, + 2 biases = 10
    W2: 2 -> 2  ->  2*2 = 4 weights, + 2 biases = 6
    Total: 10 + 6 = 16  ✅

We use tanh activation (smooth, bounded, similar in spirit to quantum measurement which
also produces bounded values in [-1, +1]).
"""

import numpy as np

INPUT_DIM = 4   # CartPole state dimension
HIDDEN_DIM = 2  # tiny on purpose, to match quantum capacity
OUTPUT_DIM = 2  # 2 actions

# Total params:  (4*2 + 2)  +  (2*2 + 2)  =  10 + 6  =  16  ✅


def init_params(seed: int = 0):
    """Initialize a flat parameter vector of length 16."""
    rng = np.random.default_rng(seed)
    # Xavier-ish init for tanh
    W1 = rng.normal(scale=0.5, size=(INPUT_DIM, HIDDEN_DIM))   # 4x2 = 8
    b1 = np.zeros(HIDDEN_DIM)                                  # 2
    W2 = rng.normal(scale=0.5, size=(HIDDEN_DIM, OUTPUT_DIM))  # 2x2 = 4
    b2 = np.zeros(OUTPUT_DIM)                                  # 2
    # Flatten to a single vector so the optimizer treats them uniformly.
    return np.concatenate([W1.flatten(), b1, W2.flatten(), b2])


def unpack(params):
    """Unpack the flat parameter vector into matrices."""
    idx = 0
    W1 = params[idx:idx + INPUT_DIM * HIDDEN_DIM].reshape(INPUT_DIM, HIDDEN_DIM)
    idx += INPUT_DIM * HIDDEN_DIM
    b1 = params[idx:idx + HIDDEN_DIM]
    idx += HIDDEN_DIM
    W2 = params[idx:idx + HIDDEN_DIM * OUTPUT_DIM].reshape(HIDDEN_DIM, OUTPUT_DIM)
    idx += HIDDEN_DIM * OUTPUT_DIM
    b2 = params[idx:idx + OUTPUT_DIM]
    return W1, b1, W2, b2


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def action_probs(state, params):
    """Forward pass: state -> action probabilities."""
    W1, b1, W2, b2 = unpack(params)
    h = np.tanh(state @ W1 + b1)            # hidden layer activations
    logits = h @ W2 + b2                    # raw output scores
    return softmax(logits)


def count_parameters():
    return INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM