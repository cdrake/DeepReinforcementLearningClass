# Speed-Optimized A* Search
#
# Key optimizations:
# 1. Tuple-based states: NPuzzleState uses numpy arrays (9 elements) for tiles,
#    but numpy hash/eq/copy have massive per-call overhead for small arrays.
#    We convert to tuples internally — native Python hash/eq is ~10x faster.
# 2. Inlined tile swap: Instead of calling env.state_action_dynamics() which does
#    np.stack, np.where, array.copy, fancy indexing for each transition, we
#    pre-extract the swap table and do a direct tuple swap (~50x faster).
# 3. torch.jit.trace with cross-call caching (avoids ~17ms trace per call).
# 4. torch.inference_mode() over no_grad().
# 5. Heuristic cache keyed on tuples.
# 6. Phase 1 greedy via Q-values, Phase 2 weighted A* fallback.
# 7. Batched nnet calls with one-hot encoding done in numpy.

from typing import List, Tuple, Dict, Optional
from environments.environment_abstract import Environment, State
import torch
from torch import nn
import numpy as np
import heapq

# Cache JIT-traced models across search() calls.
_traced_nnet_cache: Dict[int, nn.Module] = {}


def search(env: Environment, state_start: State, nnet: nn.Module) -> Optional[List[int]]:
    """ Find paths from start state to goal using trained DQN

    :param env: environment
    :param state_start: starting state
    :param nnet: trained DQN

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    """
    if env.is_terminal(state_start):
        return []

    # --- Extract puzzle-specific data for inlined transitions ---
    # Pre-compute swap table: swap_table[zero_pos][action] = swap_pos
    swap_zero_idxs = env.swap_zero_idxs  # shape (9, 4)
    num_tiles = swap_zero_idxs.shape[0]
    goal_tuple = tuple(range(num_tiles))
    one_hot = np.eye(num_tiles, dtype=np.float32)

    # Convert start state to tuple
    start_t = tuple(int(x) for x in state_start.tiles)

    # Pre-build swap table as nested tuple for fastest access
    swap_table = tuple(
        tuple(int(swap_zero_idxs[z, a]) for a in range(4))
        for z in range(num_tiles)
    )

    # Find zero position helper — store in each tuple-state as (tiles_tuple, zero_pos)
    # to avoid scanning for zero on every transition
    start_zero = start_t.index(0)

    def apply_action(tiles, zero_pos, action):
        """Inline tile swap using tuples. Returns (new_tiles, new_zero_pos)."""
        swap_pos = swap_table[zero_pos][action]
        if swap_pos == zero_pos:
            # No-op move (against wall)
            return tiles, zero_pos
        # Swap zero and target tile via list conversion
        lst = list(tiles)
        lst[zero_pos] = lst[swap_pos]
        lst[swap_pos] = 0
        return tuple(lst), swap_pos

    # --- JIT trace nnet ---
    nnet_id = id(nnet)
    if nnet_id not in _traced_nnet_cache:
        try:
            sample_input = one_hot[list(start_t)].reshape(1, -1)
            _traced_nnet_cache[nnet_id] = torch.jit.trace(
                nnet, torch.from_numpy(sample_input)
            )
        except Exception:
            _traced_nnet_cache[nnet_id] = nnet
    traced_nnet = _traced_nnet_cache[nnet_id]

    # --- Heuristic with cache ---
    h_cache: Dict[tuple, float] = {}

    def tuples_to_nnet_input(tile_tuples):
        """Convert list of tile tuples to nnet input (one-hot encoded)."""
        states_np = np.array(tile_tuples, dtype=np.intp)
        return one_hot[states_np].reshape(len(tile_tuples), -1)

    def heuristic_batch(tile_tuples):
        """Batch h-value computation with cache."""
        results = [None] * len(tile_tuples)
        missing_indices = []
        for i, t in enumerate(tile_tuples):
            h = h_cache.get(t)
            if h is not None:
                results[i] = h
            else:
                missing_indices.append(i)

        if missing_indices:
            missing = [tile_tuples[i] for i in missing_indices]
            nnet_input = tuples_to_nnet_input(missing)
            with torch.inference_mode():
                q_vals = traced_nnet(torch.from_numpy(nnet_input)).numpy()
            h_vals = -np.max(q_vals, axis=1)
            for j, idx in enumerate(missing_indices):
                h_val = float(h_vals[j])
                h_cache[missing[j]] = h_val
                results[idx] = h_val

        return results

    h_start = heuristic_batch([start_t])[0]

    # ---- Phase 1: Greedy via Q-values ----
    greedy_visited = {start_t}
    greedy_path_actions = []
    current_t = start_t
    current_z = start_zero
    max_greedy_cost = 1.5 * h_start + 3

    for _ in range(300):
        if len(greedy_path_actions) > max_greedy_cost:
            break

        # Single nnet call for Q-values (untraced is faster for batch=1)
        nnet_input = tuples_to_nnet_input([current_t])
        with torch.inference_mode():
            q_vals = nnet(torch.from_numpy(nnet_input)).numpy()[0]
        h_cache[current_t] = float(-np.max(q_vals))

        action_order = sorted(range(4), key=lambda a: q_vals[a], reverse=True)

        moved = False
        for action in action_order:
            next_t, next_z = apply_action(current_t, current_z, action)

            if next_t == current_t:  # no-op (wall)
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

    # ---- Phase 2: Weighted A* fallback (W=5.0) ----
    W = 5.0

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
                if next_t == tiles:  # no-op
                    continue
                new_g = g + 1.0  # each move costs 1

                entry = info.get(next_t)
                if entry is not None and new_g >= entry[0]:
                    continue
                info[next_t] = (new_g, tiles, zero_pos, action)

                if next_t == goal_tuple:
                    # Reconstruct path
                    actions_list = []
                    s = next_t
                    while s != start_t:
                        _, ps, pz, a = info[s]
                        actions_list.append(a)
                        s = ps
                    actions_list.reverse()
                    return actions_list

                all_candidates.append((next_t, next_z, new_g, tiles))

        if all_candidates:
            candidate_tuples = [c[0] for c in all_candidates]
            h_vals = heuristic_batch(candidate_tuples)
            for j, (next_t, next_z, new_g, par_t) in enumerate(all_candidates):
                entry = info.get(next_t)
                if entry is not None and new_g > entry[0]:
                    continue
                counter += 1
                f = new_g + W * h_vals[j]
                heapq.heappush(open_list, (f, -new_g, counter, next_t, next_z))

    return None
