# Speed-Optimized A* Search
#
# Starting from a standard batched A* (batch=16) with h(s) = -max(Q(s,a)),
# which ran at ~8.4s, 100% solved, ~91% optimal, we applied the following
# optimizations to maximize speed while keeping 100% solved and >=50% optimal:
#
# 1. torch.jit.trace with cross-call caching: JIT-compile the nnet once and
#    reuse it across all search() invocations via a module-level cache keyed
#    on model id. Profiling showed JIT tracing costs ~17ms per call -- at 149
#    test states, that was 2.5s of pure overhead before caching.
#
# 2. torch.inference_mode(): Replaced torch.no_grad() for a small additional
#    speedup on inference calls (disables more autograd bookkeeping).
#
# 3. Heuristic cache (Dict[State, float]): Shared across both search phases.
#    Avoids redundant nnet calls for states seen as successors from multiple
#    parents. The batched heuristic function checks the cache first and only
#    sends cache-misses through the nnet.
#
# 4. Phase 1 - Greedy via Q-values: Before running A*, attempt a greedy walk
#    using Q(s,a) directly to pick actions. One nnet call on the current state
#    gives Q-values for all actions; we try actions in descending Q-value order
#    and take the first one leading to an unvisited state. This avoids expanding
#    all successors (saving ~3 dynamics calls per step vs the h-based greedy).
#    A quality gate rejects greedy solutions whose path cost exceeds
#    1.5 * h(start) + 3, forcing those states into the A* phase.
#
# 5. Phase 2 - Weighted A* (W=5.0): For states where greedy fails or produces
#    a path that's too long, fall back to weighted A* with f = g + 5*h. High
#    weight aggressively steers search toward the goal, expanding far fewer
#    nodes than standard A*. Tested W values from 1.0 to 10.0:
#      W=1.0: ~8.4s, ~91% optimal (baseline)
#      W=1.5: ~7.5s, ~85% optimal
#      W=2.0: ~6.3s, ~80% optimal
#      W=3.0: ~5.2s, ~73% optimal
#      W=5.0: ~4.7s, ~69% optimal  <-- chosen
#      W=10:  ~4.6s, ~68% optimal (diminishing returns)
#
# 6. Combined info dict: Merged best_g and parent dicts into a single
#    info[state] = (best_g, parent_state, action) to halve dict lookups.
#
# Tested batch sizes (16 vs 64 -- 64 was slower due to over-expanding)
# and greedy step limits (50, 100, 300 -- 300 with quality gate was best).
# Profiling revealed JIT trace overhead was the single biggest bottleneck
# (~2.5s), and that traced inference is only faster for batches >=16 (for
# single-state calls in greedy, untraced is actually faster).
#
# Final result: ~2.3s avg, 100% solved, ~68% optimal (down from 8.4s).

from typing import List, Tuple, Dict, Optional
from environments.environment_abstract import Environment, State
import torch
from torch import nn
import numpy as np
import heapq

# [OPT-1] Cache JIT-traced models across search() calls.
# Tracing costs ~17ms and search() is called ~149 times = 2.5s of overhead.
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

    # Cache attribute lookups to avoid repeated getattr overhead
    states_to_input = env.states_to_nnet_input
    is_terminal = env.is_terminal
    get_actions = env.get_actions
    dynamics = env.state_action_dynamics

    # [OPT-1] JIT-trace the nnet once, reuse across all search() calls
    nnet_id = id(nnet)
    if nnet_id not in _traced_nnet_cache:
        try:
            sample_input = torch.from_numpy(
                np.asarray(states_to_input([state_start]), dtype=np.float32)
            )
            _traced_nnet_cache[nnet_id] = torch.jit.trace(nnet, sample_input)
        except Exception:
            _traced_nnet_cache[nnet_id] = nnet
    traced_nnet = _traced_nnet_cache[nnet_id]

    # [OPT-3] Heuristic cache -- avoids redundant nnet calls for repeated states
    h_cache: Dict[State, float] = {}

    def heuristic_batch(states):
        """Batch h-value computation with cache. Only sends cache-misses to nnet."""
        results = [None] * len(states)
        missing_indices = []
        for i, s in enumerate(states):
            h = h_cache.get(s)  # [OPT-3] check cache first
            if h is not None:
                results[i] = h
            else:
                missing_indices.append(i)

        if missing_indices:
            missing_states = [states[i] for i in missing_indices]
            nnet_input = np.asarray(states_to_input(missing_states), dtype=np.float32)
            with torch.inference_mode():  # [OPT-2] inference_mode > no_grad
                q_vals = traced_nnet(torch.from_numpy(nnet_input)).numpy()
            h_vals = -np.max(q_vals, axis=1)
            for j, idx in enumerate(missing_indices):
                h_cache[states[idx]] = float(h_vals[j])  # [OPT-3] populate cache
                results[idx] = float(h_vals[j])

        return results

    h_start = heuristic_batch([state_start])[0]

    # ---- [OPT-4] Phase 1: Greedy via Q-values ----
    # Use Q(s,a) directly to rank actions (1 nnet call per step).
    # Only calls dynamics on actions in Q-value order until an unvisited
    # successor is found, saving ~3 dynamics calls/step vs expanding all.
    # Quality gate: reject greedy solutions with path cost > 1.5 * h(start) + 3.
    greedy_visited = {state_start}
    greedy_path_actions = []
    current = state_start
    max_greedy_cost = 1.5 * h_start + 3

    for _ in range(300):
        if len(greedy_path_actions) > max_greedy_cost:
            break  # quality gate: path too long, fall through to A*

        # [OPT-4] Single nnet call gives Q-values for all actions at once
        nnet_input = np.asarray(states_to_input([current]), dtype=np.float32)
        with torch.inference_mode():  # [OPT-2]
            q_vals = traced_nnet(torch.from_numpy(nnet_input)).numpy()[0]

        actions = get_actions(current)
        action_order = sorted(actions, key=lambda a: q_vals[a], reverse=True)

        moved = False
        for action in action_order:
            reward, next_states, probs = dynamics(current, action)
            next_state = next_states[0] if len(next_states) == 1 else next_states[np.argmax(probs)]

            if next_state in greedy_visited:
                continue

            if is_terminal(next_state):
                greedy_path_actions.append(action)
                if len(greedy_path_actions) <= max_greedy_cost:
                    return greedy_path_actions  # quality gate passed
                moved = False
                break  # quality gate failed, fall through to A*

            greedy_path_actions.append(action)
            greedy_visited.add(next_state)
            current = next_state
            moved = True
            break  # took best unvisited action, next greedy step

        if not moved:
            break  # dead end or quality gate failed

    # ---- [OPT-5] Phase 2: Weighted A* fallback (W=5.0) ----
    # f = g + W*h aggressively steers toward goal, expanding far fewer nodes.
    W = 5.0

    # [OPT-6] Combined dict: info[state] = (best_g, parent_state, action)
    # Merges best_g and parent tracking into one dict to halve lookups.
    info: Dict[State, Tuple[float, Optional[State], int]] = {state_start: (0.0, None, -1)}

    counter = 0
    open_list = [(W * h_start, -0.0, counter, state_start)]
    BATCH_SIZE = 16  # tested 64 -- slower due to over-expanding

    while open_list:
        # Pop up to BATCH_SIZE nodes to expand together (amortizes nnet calls)
        batch_nodes = []
        while open_list and len(batch_nodes) < BATCH_SIZE:
            _, neg_g, _, state = heapq.heappop(open_list)
            g = -neg_g
            entry = info.get(state)
            if entry is None or g > entry[0]:
                continue  # stale entry, skip
            batch_nodes.append((g, state))

        if not batch_nodes:
            continue

        # Expand all batch nodes, collect successor candidates for one nnet call
        all_candidates = []
        for g, state in batch_nodes:
            for action in get_actions(state):
                reward, next_states, probs = dynamics(state, action)
                next_state = next_states[0] if len(next_states) == 1 else next_states[np.argmax(probs)]
                new_g = g - reward

                entry = info.get(next_state)
                if entry is not None and new_g >= entry[0]:
                    continue  # already have equal or better path
                info[next_state] = (new_g, state, action)  # [OPT-6]

                if is_terminal(next_state):
                    # Reconstruct path via parent pointers
                    actions_list = []
                    s = next_state
                    while s is not state_start:
                        _, ps, a = info[s]
                        actions_list.append(a)
                        s = ps
                    actions_list.reverse()
                    return actions_list

                all_candidates.append((action, next_state, new_g, state))

        if all_candidates:
            # [OPT-3] Batch heuristic call with cache for all candidates at once
            candidate_states = [c[1] for c in all_candidates]
            h_vals = heuristic_batch(candidate_states)
            for j, (action, next_state, new_g, par_state) in enumerate(all_candidates):
                entry = info.get(next_state)
                if entry is not None and new_g > entry[0]:
                    continue  # another candidate in batch found a better path
                info[next_state] = (new_g, par_state, action)  # [OPT-6]
                counter += 1
                f = new_g + W * h_vals[j]  # [OPT-5] weighted f-value
                heapq.heappush(open_list, (f, -new_g, counter, next_state))

    return None
