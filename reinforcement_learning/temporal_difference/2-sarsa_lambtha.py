#!/usr/bin/env python3
"""Performs the SARSA(lambtha) algorithm"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Q: numpy.ndarray containing the q-table
    state: the current state
    epsilon: the epsilon to use for the calculation

    Returns: the next action index
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action


def sarsa_lambtha(
        env, Q, lambtha, episodes=5000, max_steps=100,
        alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
        epsilon_decay=0.05):
    """
    Performs the SARSA(lambtha) algorithm

    env: the environment instance
    Q: numpy.ndarray of shape (s,a) containing the Q table
    lambtha: the eligibility trace factor
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate
    epsilon: initial threshold for epsilon greedy
    min_epsilon: minimum value that epsilon should decay to
    epsilon_decay: decay rate for updating epsilon between episodes

    Returns: Q, the updated Q table
    """
    initial_epsilon = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility_trace = np.zeros_like(Q)

        for step in range(max_steps):
            step_result = env.step(action)
            next_state, reward, terminated, truncated, info = step_result
            next_action = epsilon_greedy(Q, next_state, epsilon)

            td_error = (
                reward
                + gamma * Q[next_state, next_action]
                - Q[state, action]
            )
            eligibility_trace[state, action] += 1

            Q = Q + alpha * td_error * eligibility_trace
            eligibility_trace = gamma * lambtha * eligibility_trace

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        decay = np.exp(-epsilon_decay * ep)
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * decay

    return Q
