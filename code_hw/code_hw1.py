from typing import List, Dict, Tuple
from environments.environment_abstract import Environment, State
from numpy.typing import NDArray
import numpy as np


def mrp(state_trans_probs: NDArray, state_rewards: NDArray, gamma: float) -> NDArray:
    """ Compute the value of every state in the Markov reward process (MRP)
    |S| represents the size of the state space
    :param state_trans_probs: a |S|x|S| matrix of state transition probabilities
    :param state_rewards: a |S|x1 matrix of expected rewards for every state
    :param gamma: discount factor
    :return: a |S|x1 matrix of state values
    """
    P: NDArray[np.float64] = np.asarray(state_trans_probs, dtype=np.float64)
    r: NDArray[np.float64] = np.asarray(state_rewards, dtype=np.float64)

    # Ensure r is a column vector (n, 1)
    r = r.reshape(-1, 1)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"state_trans_probs must be square (n,n); got {P.shape}")

    n: int = P.shape[0]

    if r.shape != (n, 1):
        raise ValueError(f"state_rewards must have shape (n,1); got {r.shape}")

    # Solve (I - gamma P) v = r
    A: NDArray[np.float64] = np.eye(n, dtype=np.float64) - gamma * P

    try:
        v: NDArray[np.float64] = np.linalg.solve(A, r)
    except np.linalg.LinAlgError:
        # Fallback for singular / nearly-singular matrices
        v = np.linalg.pinv(A) @ r

    return v


def dynamic_programming(env: Environment, states: List[State], state_values: Dict[State, float],
                        gamma: float) -> Tuple[Dict[State, float], Dict[State, List[float]]]:
    """ Perform tabular dynamic programming to exactly compute the optimal value function and obtain an optimal policy

    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param gamma: the discount factor

    @return: the state value function and policy found by value iteration
    """
    # Initialize uniform random policy
    policy: Dict[State, List[float]] = {}
    for state in states:
        num_actions = len(env.get_actions(state))
        policy[state] = [1.0 / num_actions] * num_actions

    # Policy Iteration
    max_iterations = 100
    for iteration in range(max_iterations):
        # Policy Evaluation
        state_values = _evaluate_policy(env, states, policy, state_values, gamma)

        # Policy Improvement
        policy_stable = True
        for state in states:
            if env.is_terminal(state):
                continue

            # Compute Q-values for all actions
            q_values = _compute_q_values(env, state, state_values, gamma)

            # Find best action(s)
            best_action = int(np.argmax(q_values))

            # Check if policy changed
            old_best_action = int(np.argmax(policy[state]))
            if best_action != old_best_action:
                policy_stable = False

            # Update policy to be greedy
            new_policy = [0.0] * len(q_values)
            new_policy[best_action] = 1.0
            policy[state] = new_policy

        # If policy is stable, we're done
        if policy_stable:
            break

    return state_values, policy


def _evaluate_policy(env: Environment, states: List[State], policy: Dict[State, List[float]],
                     state_values: Dict[State, float], gamma: float,
                     threshold: float = 1e-6, max_iters: int = 1000) -> Dict[State, float]:
    """Iteratively compute V^pi for given policy"""
    for iteration in range(max_iters):
        max_delta = 0.0

        for state in states:
            if env.is_terminal(state):
                state_values[state] = 0.0
                continue

            old_value = state_values[state]
            new_value = 0.0

            actions = env.get_actions(state)
            for action_idx, action in enumerate(actions):
                action_prob = policy[state][action_idx]

                # Get dynamics for this action
                reward, next_states, probs = env.state_action_dynamics(state, action)

                # Compute expected value for this action
                action_value = reward
                for next_state, prob in zip(next_states, probs):
                    action_value += gamma * prob * state_values[next_state]

                new_value += action_prob * action_value

            state_values[state] = new_value
            max_delta = max(max_delta, abs(new_value - old_value))

        # Check convergence
        if max_delta < threshold:
            break

    return state_values


def _compute_q_values(env: Environment, state: State, state_values: Dict[State, float],
                     gamma: float) -> List[float]:
    """Compute Q(s,a) for all actions using state_action_dynamics"""
    actions = env.get_actions(state)
    q_values = []

    for action in actions:
        reward, next_states, probs = env.state_action_dynamics(state, action)

        q_value = reward
        for next_state, prob in zip(next_states, probs):
            q_value += gamma * prob * state_values[next_state]

        q_values.append(q_value)

    return q_values


def model_free(env: Environment, action_values: Dict[State, List[float]], gamma: float) -> Dict[State, List[float]]:
    """ Perform model free tabular reinforcement learning to compute an action-value function
    @param env: environment
    @param action_values: dictionary that maps states to their action values (list of floats)
    @param gamma: the discount factor

    @return: the action value function
    """
    # Hyperparameters
    NUM_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 200
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    ALPHA_START = 0.1
    ALPHA_END = 0.001
    ALPHA_DECAY = 0.9995

    epsilon = EPSILON_START
    alpha = ALPHA_START

    # Training loop
    for episode in range(NUM_EPISODES):
        # Sample start state
        state = env.sample_start_states(1)[0]

        # Initialize Q-values for this state if not seen before
        if state not in action_values:
            num_actions = len(env.get_actions(state))
            action_values[state] = [0.0] * num_actions

        # Episode loop
        for step in range(MAX_STEPS_PER_EPISODE):
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Explore: random action
                actions = env.get_actions(state)
                action = actions[np.random.randint(len(actions))]
            else:
                # Exploit: greedy action
                action = int(np.argmax(action_values[state]))

            # Take action and observe next state and reward
            next_state, reward = env.sample_transition(state, action)

            # Initialize Q-values for next_state if not seen before
            if next_state not in action_values:
                num_actions = len(env.get_actions(next_state))
                action_values[next_state] = [0.0] * num_actions

            # Q-learning update
            if env.is_terminal(next_state):
                target = reward
            else:
                target = reward + gamma * max(action_values[next_state])

            action_values[state][action] += alpha * (target - action_values[state][action])

            # Move to next state
            state = next_state

            # Decay learning rate per step
            alpha = max(ALPHA_END, alpha * ALPHA_DECAY)

            # Check if terminal
            if env.is_terminal(state):
                break

        # Decay epsilon per episode
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    return action_values
