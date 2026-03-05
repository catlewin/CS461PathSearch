'''
2. Agent & Search

Movement
* Allowed directions: N, S, E, W (up, down, left, right)
* Cannot move into blocked nodes

Perception
* Agent perceives only its current node and its four cardinal neighbors
* Note: The search algorithm tracks all previously visited nodes (global state), though the agent can only move to direct neighbors

Parameters: start node, goal node, graph (search space)

Termination
* Success: Goal node reached
* Failure: All reachable nodes exhausted without finding goal — report failure

Internals
* Build search tree during traversal (each visited node added as child of its predecessor)
* Store full node visit history for animation playback
* Track paths via a came_from dictionary mapping each node to its parent
'''

from collections import deque

def _reconstruct_path(parent, start, goal):
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path if path[0] == start else []


def bfs(G, start, goal, cols):
    start_id = start[0] * cols + start[1]
    goal_id  = goal[0]  * cols + goal[1]

    parent   = {start_id: None}
    queue    = deque([start_id])
    events   = [('discover', start_id), ('visit', start_id)]  # start is immediately discovered + visited
    sequence = [start_id]  # visit-only list for path reconstruction

    if start_id == goal_id:
        return True, events, sequence, [start_id], parent

    while queue:
        curr = queue.popleft()
        events.append(('visit', curr))

        for neighbor in G.neighbors(curr):
            if neighbor not in parent:
                parent[neighbor] = curr
                events.append(('discover', neighbor))
                sequence.append(neighbor)

                if neighbor == goal_id:
                    events.append(('visit', neighbor))
                    return True, events, sequence, _reconstruct_path(parent, start_id, goal_id), parent

                queue.append(neighbor)

    return False, events, sequence, [], parent


def dfs(G, start, goal, cols):
    start_id = start[0] * cols + start[1]
    goal_id  = goal[0]  * cols + goal[1]

    parent   = {start_id: None}
    stack    = [start_id]
    sequence = []

    while stack:
        curr = stack.pop()

        if curr in sequence:
            continue

        sequence.append(curr)

        if curr == goal_id:
            return True, sequence, _reconstruct_path(parent, start_id, goal_id), parent

        for neighbor in G.neighbors(curr):
            if neighbor not in parent:
                parent[neighbor] = curr
                stack.append(neighbor)

    return False, sequence, [], parent
