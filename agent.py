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
    sequence = [start_id]

    if start_id == goal_id:
        return True, sequence, [start_id], parent

    while queue:
        curr = queue.popleft()

        for neighbor in G.neighbors(curr):
            if neighbor not in parent:
                parent[neighbor] = curr
                sequence.append(neighbor)

                if neighbor == goal_id:
                    return True, sequence, _reconstruct_path(parent, start_id, goal_id), parent

                queue.append(neighbor)

    return False, sequence, [], parent


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

'''
# Run Code (debug)

from environment_generation import create_grid, generate_obstacles, get_start_and_goal, grid_to_graph

grid = generate_obstacles(create_grid())
start, goal = get_start_and_goal(grid)

G = grid_to_graph(grid)
cols = len(grid[0])

bfs_found, bfs_sequence, bfs_path, bfs_parent = bfs(G, start, goal, cols)
bfs_found, bfs_sequence, dfs_path, dfs_parent = dfs(G, start, goal, cols)

print(f"Start: {start}, Goal: {goal}")
print(f"BFS path: {bfs_path}")
print(f"DFS path: {dfs_path}")
'''