from environments.environment_abstract import Environment, State
from torch import nn
import torch
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 81, num_actions: int = 4):
        super().__init__()
        # Cribbed from run_hw3.py
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def greedy_action(env: Environment, state: State, nnet: nn.Module) -> int:
    """
    @param env: Environment
    @param state: the state
    @param nnet: neural network

    @return: a greedy action
    """
    nnet_input = env.states_to_nnet_input([state])
    x = torch.tensor(nnet_input, dtype=torch.float32)
    with torch.no_grad():
        q_values = nnet(x)
    return int(torch.argmax(q_values, dim=1).item())


def deep_rl(env: Environment) -> nn.Module:
    """
    @param env: environment
    @return: the trained neural network
    """
    q_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=0.001)
    gamma = 0.99
    batch_size = 256
    num_outer = 80
    states_per_iter = 500
    epochs_per_iter = 10

    for outer in range(num_outer):
        # Start with easy states, gradually increase difficulty (from beginning of class discussion 2/23)
        max_scramble = min(5 + outer * 2, 100)
        states = env.generate_states(states_per_iter, (0, max_scramble))

        # Collect all transitions for all states and all actions
        s_list = []
        a_list = []
        r_list = []
        ns_list = []
        d_list = []

        for state in states:
            if env.is_terminal(state):
                continue
            for action in range(4):
                ns, r = env.sample_transition(state, action)
                s_list.append(state)
                a_list.append(action)
                r_list.append(r)
                ns_list.append(ns)
                d_list.append(env.is_terminal(ns))

        if len(s_list) == 0:
            continue

        # Convert to tensors once (batch conversion is efficient)
        s_t = torch.tensor(env.states_to_nnet_input(s_list), dtype=torch.float32)
        ns_t = torch.tensor(env.states_to_nnet_input(ns_list), dtype=torch.float32)
        a_t = torch.tensor(a_list, dtype=torch.long)
        r_t = torch.tensor(r_list, dtype=torch.float32)
        d_t = torch.tensor(d_list, dtype=torch.float32)

        n = len(s_list)

        # Multiple training epochs on this batch of data
        q_net.train()
        for epoch in range(epochs_per_iter):
            # Compute targets using target network
            with torch.no_grad():
                next_q = target_net(ns_t).max(dim=1).values
                targets = r_t + gamma * next_q * (1.0 - d_t)

            # Shuffle and train in mini-batches
            indices = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = indices[i:i + batch_size]
                q_values = q_net(s_t[idx]).gather(1, a_t[idx].unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(q_values, targets[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network each outer iteration
        target_net.load_state_dict(q_net.state_dict())

    q_net.eval()
    return q_net
