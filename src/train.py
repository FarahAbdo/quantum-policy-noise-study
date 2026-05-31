"""
REINFORCE training loop — works for both quantum and classical policies.

REINFORCE is the simplest policy gradient algorithm:
    1. Roll out a full episode using the current policy.
    2. Compute the return G_t (sum of future rewards) for each timestep.
    3. Update parameters in the direction that makes high-return actions more likely:
           gradient = sum_t  G_t * grad_theta  log pi(a_t | s_t)

Why REINFORCE for this project?
    - It's the canonical baseline. Most QRL papers use it.
    - It exposes the gradient through the policy clearly — perfect for explaining
      parameter-shift vs. backprop in the interview.
    - It works on small problems like CartPole.
"""

import numpy as np
import gymnasium as gym
from tqdm import tqdm

# We use PennyLane's autograd-wrapped numpy for the quantum side so gradients flow.
from pennylane import numpy as pnp


# ---- Hyperparameters (single source of truth) ----
LEARNING_RATE = 0.01
GAMMA = 0.99              # discount factor for future rewards
MAX_EPISODES = 500        # training budget per seed
MAX_STEPS_PER_EP = 500    # CartPole-v1 caps at 500 anyway


def run_episode(policy_fn, params, env, render=False):
    """
    Run one full CartPole episode using the given policy.

    Args:
        policy_fn: callable(state, params) -> action probability vector
        params:    current policy parameters
        env:       a Gymnasium CartPole environment

    Returns:
        states, actions, rewards (lists, length = episode length)
    """
    state, _ = env.reset()
    states, actions, rewards = [], [], []

    for _ in range(MAX_STEPS_PER_EP):
        probs = policy_fn(state, params)
        # Sample an action according to the policy's probabilities (stochastic policy).
        action = int(np.random.choice(len(probs), p=probs))

        next_state, reward, terminated, truncated, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        if terminated or truncated:
            break

    return states, actions, rewards


def discounted_returns(rewards, gamma=GAMMA):
    """Compute G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ..."""
    G = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    # Normalize returns — this is a standard variance-reduction trick.
    # It centers the gradient signal around zero so the optimizer behaves better.
    if len(G) > 1 and G.std() > 1e-8:
        G = (G - G.mean()) / (G.std() + 1e-8)
    return G


# ====================================================================
# CLASSICAL TRAINING
# ====================================================================

def train_classical(seed: int, max_episodes: int = MAX_EPISODES, verbose: bool = True):
    """
    Train the classical MLP policy with REINFORCE.
    Uses analytic gradients (we derive them by hand — it's only 16 params).

    Returns:
        params: trained parameter vector
        episode_rewards: list of total reward per episode
    """
    from src import classical_policy as cp

    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    params = cp.init_params(seed=seed)
    episode_rewards = []

    iterator = tqdm(range(max_episodes), desc=f"Classical seed={seed}") if verbose else range(max_episodes)

    for ep in iterator:
        states, actions, rewards = run_episode(cp.action_probs, params, env)
        G = discounted_returns(rewards)

        # Compute the policy gradient by hand.
        # For softmax policy:  d log pi(a|s) / d logits  =  one_hot(a) - softmax(logits)
        # Then backprop through the MLP layers (chain rule).
        grad = np.zeros_like(params)
        for s, a, g in zip(states, actions, G):
            probs = cp.action_probs(s, params)
            dlogits = -probs
            dlogits[a] += 1.0                        # one_hot(a) - probs

            # Backprop through the MLP analytically
            W1, b1, W2, b2 = cp.unpack(params)
            h = np.tanh(s @ W1 + b1)

            dW2 = np.outer(h, dlogits)               # 2x2
            db2 = dlogits                            # 2
            dh = (W2 @ dlogits) * (1 - h ** 2)       # 2 (tanh' = 1 - tanh^2)
            dW1 = np.outer(s, dh)                    # 4x2
            db1 = dh                                 # 2

            flat = np.concatenate([dW1.flatten(), db1, dW2.flatten(), db2])
            grad += g * flat

        # Gradient ASCENT (we want to maximize reward), so we ADD the grad.
        params = params + LEARNING_RATE * grad

        episode_rewards.append(sum(rewards))

    env.close()
    return params, episode_rewards


# ====================================================================
# QUANTUM TRAINING
# ====================================================================

def train_quantum(seed: int, noise_prob: float = 0.0,
                  max_episodes: int = MAX_EPISODES, verbose: bool = True):
    """
    Train the quantum variational policy with REINFORCE.
    Uses PennyLane's parameter-shift gradients automatically.

    Returns:
        params: trained parameter array
        episode_rewards: list of total reward per episode
    """
    from src import quantum_policy as qp

    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    circuit = qp.make_policy(noise_prob=noise_prob)
    params = pnp.array(qp.init_params(seed=seed), requires_grad=True)
    episode_rewards = []

    # log_prob_fn: returns log pi(a | s) so PennyLane can differentiate it.
    def log_prob(params, state, action):
        logits = pnp.array([circuit(state, params)[i] for i in range(qp.N_ACTIONS)])
        # log softmax, numerically stable
        max_l = pnp.max(logits)
        log_z = max_l + pnp.log(pnp.sum(pnp.exp(logits - max_l)))
        return logits[action] - log_z

    grad_log_prob = qml_grad_fn(log_prob)

    iterator = tqdm(range(max_episodes), desc=f"Quantum p={noise_prob} seed={seed}") if verbose else range(max_episodes)

    for ep in iterator:
        # For speed during rollout we DON'T need gradients — only for the update step.
        def policy_for_rollout(s, p):
            return qp.action_probs(circuit, s, p)

        states, actions, rewards = run_episode(policy_for_rollout, params, env)
        G = discounted_returns(rewards)

        grad = pnp.zeros_like(params)
        for s, a, g in zip(states, actions, G):
            grad = grad + g * grad_log_prob(params, s, a)

        # Gradient ascent
        params = params + LEARNING_RATE * grad

        episode_rewards.append(sum(rewards))

    env.close()
    return params, episode_rewards


def qml_grad_fn(fn):
    """Wrap PennyLane's grad to take the gradient w.r.t. the first argument only."""
    import pennylane as qml
    return qml.grad(fn, argnum=0)