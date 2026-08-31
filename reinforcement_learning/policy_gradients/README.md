# Policy Gradients

This project covers Policy Gradient methods in reinforcement learning,
including computing a softmax policy, the Monte-Carlo policy gradient
algorithm (REINFORCE), and training an agent using policy gradients.

## Tasks

### 0. Simple Policy function
File: `policy_gradient.py`

Function `policy(matrix, weight)` computes the policy with a weight
of a matrix.

- `matrix`: `numpy.ndarray` containing the state
- `weight`: `numpy.ndarray` containing the weight
- Returns: the policy (action probabilities), computed via softmax
  of the dot product of `matrix` and `weight`

### Requirements
- Ubuntu 20.04 LTS, Python 3.9
- numpy

### Author
fatmaxanim2006

### 1. Compute the Monte-Carlo policy gradient
File: `policy_gradient.py`

Function `policy_gradient(state, weight)` computes the Monte-Carlo
policy gradient based on a state and a weight matrix, using the
`policy` function.

- `state`: matrix representing the current observation of the
  environment
- `weight`: matrix of random weight
- Returns: the action and the gradient (in this order)

### 2. Implement the training
File: `train.py`

Function `train(env, nb_episodes, alpha=0.000045, gamma=0.98)`
implements a full training loop using the Monte-Carlo policy
gradient (REINFORCE) algorithm.

- `env`: initial environment
- `nb_episodes`: number of episodes used for training
- `alpha`: the learning rate
- `gamma`: the discount factor
- Returns: all values of the score (sum of all rewards during
  one episode loop)

Prints `Episode: {} Score: {}` after each episode.

### 3. Animate iteration
File: `train.py` (updated)

The `train` function now accepts an additional optional parameter
`show_result` (default: `False`). When set to `True`, the
environment is rendered every 1000 episodes.

- `env`: initial environment
- `nb_episodes`: number of episodes used for training
- `alpha`: the learning rate
- `gamma`: the discount factor
- `show_result`: if `True`, render the environment every 1000
  episodes
- Returns: all values of the score
