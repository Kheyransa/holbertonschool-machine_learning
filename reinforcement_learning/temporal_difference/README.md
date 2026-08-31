# Temporal Difference

This project covers Temporal Difference learning methods in
reinforcement learning, including Monte Carlo, TD(λ), and SARSA(λ)
algorithms using the OpenAI Gymnasium `FrozenLake` environment.

## Tasks

### 0. Monte Carlo
File: `0-monte_carlo.py`

Function `monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`
performs the Monte Carlo algorithm.

- `env`: environment instance
- `V`: `numpy.ndarray` of shape `(s,)` containing the value estimate
- `policy`: function that takes in a state and returns the next
  action to take
- `episodes`: total number of episodes to train over
- `max_steps`: maximum number of steps per episode
- `alpha`: learning rate
- `gamma`: discount rate
- Returns: `V`, the updated value estimate

### Requirements
- Ubuntu 20.04 LTS, Python 3.9
- gymnasium
- numpy

### Author
fatmaxanim2006

### 1. TD(λ)
File: `1-td_lambtha.py`

Function `td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`
performs the TD(λ) algorithm.

- `env`: the environment instance
- `V`: `numpy.ndarray` of shape `(s,)` containing the value estimate
- `policy`: function that takes in a state and returns the next
  action to take
- `lambtha`: the eligibility trace factor
- `episodes`: total number of episodes to train over
- `max_steps`: maximum number of steps per episode
- `alpha`: learning rate
- `gamma`: discount rate
- Returns: `V`, the updated value estimate

### 2. SARSA(λ)
File: `2-sarsa_lambtha.py`

Function `sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)`
performs the SARSA(λ) algorithm.

- `env`: the environment instance
- `Q`: `numpy.ndarray` of shape `(s,a)` containing the Q table
- `lambtha`: the eligibility trace factor
- `episodes`: total number of episodes to train over
- `max_steps`: maximum number of steps per episode
- `alpha`: learning rate
- `gamma`: discount rate
- `epsilon`: initial threshold for epsilon greedy
- `min_epsilon`: minimum value that epsilon should decay to
- `epsilon_decay`: decay rate for updating epsilon between episodes
- Returns: `Q`, the updated Q table
