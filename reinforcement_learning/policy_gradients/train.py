#!/usr/bin/env python3
"""Implements the full training for policy gradient"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98,
          show_result=False):
    """
    Implements a full training with policy gradient

    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    show_result: if True, renders the environment every 1000
                 episodes

    Returns: all values of the score (sum of all rewards during
             one episode loop)
    """
    weight = np.random.rand(
        env.observation_space.shape[0], env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        score = 0

        while True:
            if show_result and episode % 1000 == 0:
                env.render()

            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, info = env.step(
                action)

            grads.append(grad)
            rewards.append(reward)
            score += reward

            state = next_state

            if terminated or truncated:
                break

        for i in range(len(grads)):
            discounted_rewards = sum(
                [r * (gamma ** t) for t, r in enumerate(rewards[i:])])
            weight += alpha * grads[i] * discounted_rewards

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores
