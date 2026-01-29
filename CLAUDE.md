# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Reinforcement Learning course repository where students implement RL algorithms through homework assignments. The codebase is built on top of Grid-World-Robot by Nitish Gupta.

## Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate rlclass
```

The environment uses Python 3.10 with NumPy, PyTorch, Pillow, and Tkinter.

## Running Homework Assignments

### Homework 1: MRP and Tabular Methods

```bash
# Test MRP implementation
python run_hw1.py --task mrp --gamma 0.9

# Dynamic Programming (Policy Iteration)
python run_hw1.py --task dp --env aifarm --gamma 0.9

# Model-Free RL
python run_hw1.py --task mf --env aifarm --gamma 0.9

# Grade against expected output
python run_hw1.py --task dp --env aifarm --gamma 0.9 --grade
```

Available environments for HW1: `aifarm` (deterministic), `aifarm_0.2` (stochastic with 20% random right movement)

### Homework 2: Deep RL

```bash
# Train on standard 8-puzzle
python run_hw2.py

# Train on randomized 8-puzzle
python run_hw2.py --rand
```

### Homework 3: Search with Neural Networks

```bash
# Run search using trained DQN
python run_hw3.py

# Grade against optimal solutions
python run_hw3.py --grade
```

## Code Architecture

### Environment Abstraction

The codebase uses an abstract environment interface ([environments/environment_abstract.py](environments/environment_abstract.py)) that all environments implement:

- **State**: Abstract base class for state representation (must implement `__hash__` and `__eq__`)
- **Environment**: Defines the RL interface with methods:
  - `get_actions(state)`: Returns available actions
  - `is_terminal(state)`: Checks if state is terminal
  - `state_action_dynamics(state, action)`: Returns (reward, next_states, probabilities) for model-based methods
  - `sample_transition(state, action)`: Samples (next_state, reward) for model-free methods
  - `sample_start_states(num_states)`: Generates random start states
  - `states_to_nnet_input(states)`: Converts states to neural network input format

### Implemented Environments

**FarmGridWorld** ([environments/farm_grid_world.py](environments/farm_grid_world.py)):
- Grid world where agent navigates to goal while avoiding plants (-50 reward) and rocks (-10 reward)
- Each move costs -1 reward
- Can be stochastic (with probability `rand_right`, actions randomly become "move right")
- Maps loaded from [maps/](maps/) directory
- Visualization available via [visualizer/farm_visualizer.py](visualizer/farm_visualizer.py)

**NPuzzle** ([environments/n_puzzle.py](environments/n_puzzle.py)):
- Standard 8-puzzle sliding tile puzzle (3x3 grid)
- Actions: Up, Down, Left, Right (moves tile into blank space)
- Each action costs -1 reward
- Can be deterministic or randomized (randomized transition probabilities)
- Pre-computed swap indices for efficient state transitions

### Student Implementation Structure

Students implement algorithms in `code_hw/code_hw*.py` files:

**code_hw1.py**:
- `mrp(state_trans_probs, state_rewards, gamma)`: Solve Markov Reward Process
- `dynamic_programming(env, states, state_values, gamma)`: Policy iteration for optimal policy
- `model_free(env, action_values, gamma)`: Model-free RL (e.g., Q-learning, SARSA)

**code_hw2.py**:
- `greedy_action(env, state, nnet)`: Select greedy action from neural network
- `deep_rl(env)`: Train neural network using deep RL

**code_hw3.py**:
- `search(env, state_start, nnet)`: Search algorithm using trained DQN as heuristic

### Environment Utilities

[environments/env_utils.py](environments/env_utils.py) provides `get_environment(env_name)` which returns (environment, visualizer, states):
- `"aifarm"` or `"aifarm_X"`: FarmGridWorld with random right probability X
- `"puzzle8"`: Deterministic 8-puzzle
- `"puzzle8_rand"`: Randomized 8-puzzle

Uses map1.txt for farm environments by default.

## Important Implementation Notes

- **State Hashing**: All State subclasses must properly implement `__hash__()` and `__eq__()` for use in dictionaries
- **Neural Network Input**: Each environment defines its own state representation via `states_to_nnet_input()`
  - FarmGridWorld: One-hot encoding of agent position (100-dim for 10x10 grid)
  - NPuzzle: One-hot encoding of all tiles (81-dim for 8-puzzle)
- **Rewards**: All environments use negative rewards (costs) where goal is to minimize cumulative cost
- **Terminal States**: Terminal state transitions return 0 reward and stay in same state
- **Grading Data**: Some homework has pickle files in `data_hw/` or `grading/` for validation
