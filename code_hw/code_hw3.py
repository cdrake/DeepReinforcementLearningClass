# A* Search — optimized for 8-puzzle, generic fallback for other environments
#
# 8-puzzle optimizations:
# 1. Tuple-based states with zero_pos tracking
# 2. Batched torch inference — single forward pass for all 181K states
# 3. Pre-computed heuristics for all 181,440 reachable states at startup
# 4. Combined h = max(manhattan, nnet_h) for better-informed search
# 5. Vectorized Manhattan distance (numpy, not per-state loop)
# 6. No successor cache — inline transitions save ~0.4s precompute
# 7. Greedy Phase 1 + Weighted A* Phase 2
#
# Generic fallback (non-puzzle environments):
# Uses only the abstract Environment interface (get_actions, sample_transition,
# is_terminal, states_to_nnet_input) and torch nnet forward pass.
# Same greedy + A* structure, with batched heuristic computation.
#
# Hyperparameter tuning results (puzzle8, seed=42, 149 states):
#
# BATCH_SIZE (A* node expansion batch):
#   1  -> 85.91% optimal, 1.79s
#   4  -> 87.25% optimal, 1.65s
#   8  -> 87.92% optimal, 1.36s
#   16 -> 87.92% optimal, 1.69s
#   32 -> 89.26% optimal, 1.74s
#   64 -> 91.28% optimal, 1.64s
#  128 -> 92.62% optimal, 2.00s
#
# W (A* heuristic weight, lower = more optimal but slower):
#   0.50 -> 97.32% optimal, 6.23s
#   0.75 -> 95.97% optimal, 2.45s
#   1.00 -> 87.92% optimal, 2.07s
#   1.50 -> 82.55% optimal, 1.51s
#   2.00 -> 75.84% optimal, 1.56s
#   3.00 -> 69.80% optimal, 1.47s
#
# greedy_mult (max greedy path = mult * h_start + add):
#   1.0 -> 89.93% optimal, 1.69s
#   1.5 -> 87.92% optimal, 1.46s
#   2.0 -> 84.56% optimal, 1.69s
#   3.0 -> 76.51% optimal, 1.50s
#
# greedy_max: No impact (greedy always terminates before 50 steps)
# CHUNK: No quality impact; 8000 marginally fastest for precomputation
#
# After removing successor cache + vectorized Manhattan (saves ~0.8s):
#   W=3.0, BS=4  -> 62.4% optimal, 0.525s  <-- selected for speed (extra credit)
#   W=2.0, BS=4  -> 71.1% optimal, 0.677s
#   W=1.5, BS=16 -> 84.6% optimal, 0.790s
#   W=0.75, BS=64 -> 95.97% optimal, ~2.4s  (quality mode)
#
# Extra credit requires >=50% optimal; optimizing for fastest time.

from typing import List, Tuple, Dict, Optional
from environments.environment_abstract import Environment, State
from environments.n_puzzle import NPuzzle
from torch import nn
import torch
import numpy as np
import heapq

# --- Module-level caches for puzzle8 (persist across search() calls) ---
_precomputed_h: Optional[Dict[tuple, float]] = None
_precomputed_q: Optional[Dict[tuple, np.ndarray]] = None
_swap_table: Optional[tuple] = None


# ============================================================
# Puzzle8-specific helpers
# ============================================================

def _precompute_all(env: Environment, nnet: nn.Module):
    """BFS all 181,440 reachable states; pre-compute h-values and Q-values."""
    global _precomputed_h, _precomputed_q, _swap_table

    swap_zero_idxs = env.swap_zero_idxs
    num_tiles = swap_zero_idxs.shape[0]
    one_hot = np.eye(num_tiles, dtype=np.float32)

    # Build swap table
    _swap_table = tuple(
        tuple(int(swap_zero_idxs[z, a]) for a in range(4))
        for z in range(num_tiles)
    )

    # BFS from goal to enumerate all reachable states
    goal = tuple(range(num_tiles))
    all_states = {goal: goal.index(0)}
    frontier = [(goal, goal.index(0))]
    while frontier:
        next_frontier = []
        for tiles, zp in frontier:
            for a in range(4):
                sp = _swap_table[zp][a]
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

    # Batched torch inference — architecture-agnostic, works with any DQN
    all_list = list(all_states.keys())
    states_np = np.array(all_list, dtype=np.intp)
    nnet_input = one_hot[states_np].reshape(len(all_list), -1)
    with torch.no_grad():
        q_all = nnet(torch.from_numpy(nnet_input).float()).numpy()
    nnet_h = -np.max(q_all, axis=1)

    # Vectorized Manhattan distance (replaces per-state loop, ~25x faster)
    dim = int(np.sqrt(num_tiles))
    goal_row = np.arange(num_tiles) // dim
    goal_col = np.arange(num_tiles) % dim
    pos_row = np.arange(num_tiles) // dim
    pos_col = np.arange(num_tiles) % dim
    tile_goal_rows = goal_row[states_np]
    tile_goal_cols = goal_col[states_np]
    manhattan_all = np.abs(tile_goal_rows - pos_row) + np.abs(tile_goal_cols - pos_col)
    manhattan_all[states_np == 0] = 0
    manhattan_sums = manhattan_all.sum(axis=1)

    # Build h and q dicts
    h_values = np.maximum(manhattan_sums.astype(np.float64), nnet_h)
    _precomputed_h = dict(zip(all_list, h_values))
    _precomputed_q = {t: q_all[j] for j, t in enumerate(all_list)}


def _search_puzzle(env: Environment, state_start: State, nnet: nn.Module) -> Optional[List[int]]:
    """Optimized A* for 8-puzzle with precomputed heuristics."""
    # --- One-time precomputation of all heuristics ---
    if _precomputed_h is None:
        _precompute_all(env, nnet)

    num_tiles = env.swap_zero_idxs.shape[0]
    goal_tuple = tuple(range(num_tiles))
    start_t = tuple(int(x) for x in state_start.tiles)
    start_zero = start_t.index(0)
    swap_table = _swap_table

    h_start = _precomputed_h[start_t]

    # ---- Phase 1: Greedy via pre-computed Q-values ----
    greedy_visited = {start_t}
    greedy_path_actions = []
    current_t = start_t
    current_z = start_zero
    max_greedy_cost = 1.0 * h_start + 3

    for _ in range(300):
        if len(greedy_path_actions) > max_greedy_cost:
            break

        q_vals = _precomputed_q[current_t]
        action_order = sorted(range(4), key=lambda a: q_vals[a], reverse=True)

        moved = False
        for action in action_order:
            sp = swap_table[current_z][action]
            if sp == current_z:
                continue
            lst = list(current_t)
            lst[current_z] = lst[sp]
            lst[sp] = 0
            next_t = tuple(lst)

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
            current_z = sp
            moved = True
            break

        if not moved:
            break

    # ---- Phase 2: Weighted A* fallback ----
    # W=3.0, BS=4: 62.4% optimal, ~0.5s (selected for speed extra credit, >=50% required)
    W = 3.0
    BATCH_SIZE = 4

    # info[tiles_tuple] = (best_g, parent_tuple, parent_zero, action)
    info: Dict[tuple, Tuple[float, Optional[tuple], int, int]] = {
        start_t: (0.0, None, start_zero, -1)
    }

    counter = 0
    open_list = [(W * h_start, -0.0, counter, start_t, start_zero)]

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
                sp = swap_table[zero_pos][action]
                if sp == zero_pos:
                    continue
                lst = list(tiles)
                lst[zero_pos] = lst[sp]
                lst[sp] = 0
                next_t = tuple(lst)
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

                all_candidates.append((next_t, sp, new_g))

        if all_candidates:
            for next_t, next_z, new_g in all_candidates:
                entry = info.get(next_t)
                if entry is not None and new_g > entry[0]:
                    continue
                counter += 1
                f = new_g + W * _precomputed_h[next_t]
                heapq.heappush(open_list, (f, -new_g, counter, next_t, next_z))

    return None


# ============================================================
# Generic search for any Environment
# ============================================================

def _nnet_heuristic(env: Environment, states: List[State], nnet: nn.Module) -> np.ndarray:
    """Compute h = -max(Q-values) for a batch of states using the nnet."""
    nnet_input = env.states_to_nnet_input(states)
    with torch.no_grad():
        q_vals = nnet(torch.from_numpy(nnet_input).float()).numpy()
    return -np.max(q_vals, axis=1)


def _search_generic(env: Environment, state_start: State, nnet: nn.Module) -> Optional[List[int]]:
    """Generic A* search using only the abstract Environment interface."""
    # Compute heuristic for start state
    h_vals = _nnet_heuristic(env, [state_start], nnet)
    h_start = float(h_vals[0])

    # ---- Phase 1: Greedy ----
    greedy_visited = {state_start}
    greedy_path_actions = []
    current = state_start
    max_greedy_steps = int(1.5 * max(h_start, 1) + 3)

    for _ in range(max_greedy_steps):
        # Get Q-values for current state
        nnet_input = env.states_to_nnet_input([current])
        with torch.no_grad():
            q_vals = nnet(torch.from_numpy(nnet_input).float()).numpy()[0]

        actions = env.get_actions(current)
        action_order = sorted(actions, key=lambda a: q_vals[a], reverse=True)

        moved = False
        for action in action_order:
            next_state, reward = env.sample_transition(current, action)
            if next_state == current:
                continue
            if next_state in greedy_visited:
                continue

            if env.is_terminal(next_state):
                greedy_path_actions.append(action)
                return greedy_path_actions

            greedy_path_actions.append(action)
            greedy_visited.add(next_state)
            current = next_state
            moved = True
            break

        if not moved:
            break

    # ---- Phase 2: A* ----
    # info[state] = (best_g, parent_state, action)
    info: Dict[State, Tuple[float, Optional[State], int]] = {
        state_start: (0.0, None, -1)
    }

    counter = 0
    open_list = [(h_start, 0.0, counter, state_start)]

    while open_list:
        _, neg_g, _, current = heapq.heappop(open_list)
        g = -neg_g
        entry = info.get(current)
        if entry is None or g > entry[0]:
            continue

        actions = env.get_actions(current)
        # Gather successors
        successors = []
        for action in actions:
            next_state, reward = env.sample_transition(current, action)
            if next_state == current:
                continue
            new_g = g + (-reward)

            entry_next = info.get(next_state)
            if entry_next is not None and new_g >= entry_next[0]:
                continue
            info[next_state] = (new_g, current, action)

            if env.is_terminal(next_state):
                # Reconstruct path
                actions_list = []
                s = next_state
                while s != state_start:
                    _, ps, a = info[s]
                    actions_list.append(a)
                    s = ps
                actions_list.reverse()
                return actions_list

            successors.append((next_state, new_g))

        # Batch heuristic computation for all successors
        if successors:
            succ_states = [s for s, _ in successors]
            h_vals = _nnet_heuristic(env, succ_states, nnet)
            for idx, (next_state, new_g) in enumerate(successors):
                # Re-check in case a better path was found while processing batch
                entry_next = info.get(next_state)
                if entry_next is not None and new_g > entry_next[0]:
                    continue
                counter += 1
                f = new_g + float(h_vals[idx])
                heapq.heappush(open_list, (f, -new_g, counter, next_state))

    return None


# ============================================================
# Main entry point
# ============================================================

def search(env: Environment, state_start: State, nnet: nn.Module) -> Optional[List[int]]:
    """Find path from start state to goal using trained DQN as heuristic."""
    if env.is_terminal(state_start):
        return []

    if isinstance(env, NPuzzle):
        return _search_puzzle(env, state_start, nnet)
    else:
        return _search_generic(env, state_start, nnet)
