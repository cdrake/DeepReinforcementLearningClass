# Combined Optimized A* Search
#
# Combines the best optimizations:
# 1. Tuple-based states with zero_pos tracking (baseline)
# 2. Pure numpy inference — eliminates all torch overhead
# 3. Pre-computed heuristics for all 181,440 reachable states at startup
# 4. Combined h = max(manhattan, nnet_h) for better-informed search
# 5. Successor cache — avoids repeated tuple->list->tuple conversion
# 6. Greedy Phase 1 + Weighted A* Phase 2 (baseline algorithm)

from typing import List, Tuple, Dict, Optional
from environments.environment_abstract import Environment, State
from torch import nn
import numpy as np
import heapq

# --- Module-level caches (persist across search() calls) ---
_numpy_weights_cache: Dict[int, tuple] = {}
_precomputed_h: Optional[Dict[tuple, float]] = None
_precomputed_q: Optional[Dict[tuple, np.ndarray]] = None
_successor_cache: Dict[tuple, tuple] = {}


def _get_numpy_weights(nnet: nn.Module) -> tuple:
    """Extract nnet weights as numpy arrays, cached by model id."""
    nnet_id = id(nnet)
    if nnet_id not in _numpy_weights_cache:
        sd = nnet.state_dict()
        W1 = sd['layers.0.0.weight'].numpy().copy()
        b1 = sd['layers.0.0.bias'].numpy().copy()
        W2 = sd['layers.1.0.weight'].numpy().copy()
        b2 = sd['layers.1.0.bias'].numpy().copy()
        W3 = sd['layers.2.0.weight'].numpy().copy()
        b3 = sd['layers.2.0.bias'].numpy().copy()
        _numpy_weights_cache[nnet_id] = (W1, b1, W2, b2, W3, b3)
    return _numpy_weights_cache[nnet_id]


def _numpy_forward(x: np.ndarray, weights: tuple) -> np.ndarray:
    """Pure numpy forward pass: Linear->ReLU->Linear->ReLU->Linear."""
    W1, b1, W2, b2, W3, b3 = weights
    x = x @ W1.T + b1
    np.maximum(x, 0, out=x)
    x = x @ W2.T + b2
    np.maximum(x, 0, out=x)
    return x @ W3.T + b3


def _manhattan_distance(tiles: tuple) -> int:
    """Sum of Manhattan distances of each tile to its goal position."""
    dist = 0
    for pos, tile in enumerate(tiles):
        if tile == 0:
            continue
        dist += abs(tile // 3 - pos // 3) + abs(tile % 3 - pos % 3)
    return dist


def _precompute_all(env: Environment, nnet: nn.Module):
    """BFS all 181,440 reachable states; pre-compute h-values and Q-values."""
    global _precomputed_h, _precomputed_q

    swap_zero_idxs = env.swap_zero_idxs
    num_tiles = swap_zero_idxs.shape[0]
    weights = _get_numpy_weights(nnet)
    one_hot = np.eye(num_tiles, dtype=np.float32)

    # Build swap table
    swap_table = tuple(
        tuple(int(swap_zero_idxs[z, a]) for a in range(4))
        for z in range(num_tiles)
    )

    # BFS from goal to enumerate all reachable states
    goal = tuple(range(num_tiles))
    all_states = {goal: goal.index(0)}  # tiles -> zero_pos
    frontier = [(goal, goal.index(0))]
    while frontier:
        next_frontier = []
        for tiles, zp in frontier:
            for a in range(4):
                sp = swap_table[zp][a]
                if sp == zp:
                    continue
                lst = list(tiles)
                lst[zp] = lst[sp]
                lst[sp] = 0
                nt = tuple(lst)
                if nt not in all_states:
                    all_states[nt] = sp
                    next_frontier.append((nt, sp))
        frontier = next_frontier

    # Batch nnet inference with numpy (no torch overhead)
    all_list = list(all_states.keys())
    _precomputed_h = {}
    _precomputed_q = {}
    CHUNK = 2000
    for i in range(0, len(all_list), CHUNK):
        chunk = all_list[i:i + CHUNK]
        states_np = np.array(chunk, dtype=np.intp)
        nnet_input = one_hot[states_np].reshape(len(chunk), -1)
        q_vals = _numpy_forward(nnet_input, weights)
        nnet_h = -np.max(q_vals, axis=1)
        for j, t in enumerate(chunk):
            m = _manhattan_distance(t)
            _precomputed_h[t] = max(m, float(nnet_h[j]))
            _precomputed_q[t] = q_vals[j]

    # Also populate successor cache for all states
    for tiles, zp in all_states.items():
        for a in range(4):
            sp = swap_table[zp][a]
            if sp == zp:
                _successor_cache[(tiles, a)] = (tiles, zp)
            else:
                lst = list(tiles)
                lst[zp] = lst[sp]
                lst[sp] = 0
                _successor_cache[(tiles, a)] = (tuple(lst), sp)


def search(env: Environment, state_start: State, nnet: nn.Module) -> Optional[List[int]]:
    """Find paths from start state to goal using trained DQN + Manhattan heuristic."""
    if env.is_terminal(state_start):
        return []

    # --- One-time precomputation of all heuristics ---
    if _precomputed_h is None:
        _precompute_all(env, nnet)

    num_tiles = env.swap_zero_idxs.shape[0]
    goal_tuple = tuple(range(num_tiles))
    start_t = tuple(int(x) for x in state_start.tiles)
    start_zero = start_t.index(0)

    def apply_action(tiles, zero_pos, action):
        """O(1) successor lookup from pre-computed cache."""
        return _successor_cache[(tiles, action)]

    def h(tiles):
        return _precomputed_h[tiles]

    h_start = h(start_t)

    # ---- Phase 1: Greedy via pre-computed Q-values ----
    greedy_visited = {start_t}
    greedy_path_actions = []
    current_t = start_t
    current_z = start_zero
    max_greedy_cost = 1.5 * h_start + 3

    for _ in range(300):
        if len(greedy_path_actions) > max_greedy_cost:
            break

        q_vals = _precomputed_q[current_t]
        action_order = sorted(range(4), key=lambda a: q_vals[a], reverse=True)

        moved = False
        for action in action_order:
            next_t, next_z = apply_action(current_t, current_z, action)
            if next_t == current_t:
                continue
            if next_t in greedy_visited:
                continue

            if next_t == goal_tuple:
                greedy_path_actions.append(action)
                if len(greedy_path_actions) <= max_greedy_cost:
                    return greedy_path_actions
                moved = False
                break

            greedy_path_actions.append(action)
            greedy_visited.add(next_t)
            current_t = next_t
            current_z = next_z
            moved = True
            break

        if not moved:
            break

    # ---- Phase 2: Weighted A* fallback ----
    W = 1.0

    # info[tiles_tuple] = (best_g, parent_tuple, parent_zero, action)
    info: Dict[tuple, Tuple[float, Optional[tuple], int, int]] = {
        start_t: (0.0, None, start_zero, -1)
    }

    counter = 0
    open_list = [(W * h_start, -0.0, counter, start_t, start_zero)]
    BATCH_SIZE = 16

    while open_list:
        batch_nodes = []
        while open_list and len(batch_nodes) < BATCH_SIZE:
            _, neg_g, _, tiles, zero_pos = heapq.heappop(open_list)
            g = -neg_g
            entry = info.get(tiles)
            if entry is None or g > entry[0]:
                continue
            batch_nodes.append((g, tiles, zero_pos))

        if not batch_nodes:
            continue

        all_candidates = []
        for g, tiles, zero_pos in batch_nodes:
            for action in range(4):
                next_t, next_z = apply_action(tiles, zero_pos, action)
                if next_t == tiles:
                    continue
                new_g = g + 1.0

                entry = info.get(next_t)
                if entry is not None and new_g >= entry[0]:
                    continue
                info[next_t] = (new_g, tiles, zero_pos, action)

                if next_t == goal_tuple:
                    actions_list = []
                    s = next_t
                    while s != start_t:
                        _, ps, pz, a = info[s]
                        actions_list.append(a)
                        s = ps
                    actions_list.reverse()
                    return actions_list

                all_candidates.append((next_t, next_z, new_g))

        if all_candidates:
            for next_t, next_z, new_g in all_candidates:
                entry = info.get(next_t)
                if entry is not None and new_g > entry[0]:
                    continue
                counter += 1
                f = new_g + W * h(next_t)
                heapq.heappush(open_list, (f, -new_g, counter, next_t, next_z))

    return None
