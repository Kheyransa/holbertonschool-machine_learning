# Q-Learning

This project covers Q-learning and reinforcement learning basics using
the OpenAI Gymnasium `FrozenLake-v1` environment.

## Tasks

### 0. Load the Environment
File: `0-load_env.py`

Function `load_frozen_lake(desc=None, map_name=None, is_slippery=False)`
loads the pre-made `FrozenLakeEnv` environment from `gymnasium`.

- `desc`: `None` or a list of lists containing a custom description of
  the map to load for the environment
- `map_name`: `None` or a string containing the pre-made map to load
- If both `desc` and `map_name` are `None`, the environment loads a
  randomly generated 8x8 map
- `is_slippery`: boolean to determine if the ice is slippery
- Returns: the environment

### Requirements
- Ubuntu 20.04 LTS, Python 3.9
- gymnasium

### Author
fatmaxanim2006
