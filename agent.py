"""
agent.py
--------
Implements BFS & DFS uninformed search agents for grid-based pathfinding
using a NetworkX graph representation of a 2D grid. Both algorithms track
visitation history and reconstruct the path from start to goal upon completion.

Functions
---------
_reconstruct_path(parent, start, goal)
    Helper function to reconstruct the path from start to goal.

bfs(G, start, goal, cols)
    Breadth-first search returning events, visitation sequence,
    path, and parent map.

dfs(G, start, goal, cols)
    Depth-first search returning visitation sequence, path,
    and parent map.
"""

from collections import deque

def _reconstruct_path(parent, start, goal):
    '''
    Reconstruct the path from start to goal.

    :param parent: Dictionary mapping each discovered node ID to its parent node ID. Used for path reconstruction and tree visualization.
    :param start: tuple coordinates of the start cell.
    :param goal: tuple coordinates of the goal cell.

    :return:- path (list[int]): Node IDs forming a path from start to goal inclusive. Empty list if no path was found.
    '''

    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path if path[0] == start else []

def bfs(G, start, goal, cols):
    """
    Breadth-first search of a grid graph from start to goal.

    Parameters
    ----------
    G : networkx.Graph
        Grid graph where nodes are flat integer IDs (r * cols + c) and
        edges connect valid cardinal neighbors.
    start : tuple
        (row, col) coordinates of the start cell.
    goal : tuple
        (row, col) coordinates of the goal cell.
    cols : int
        Number of columns in the grid, used to convert (row, col) to node ID.

    Returns
    -------
    tuple: A 5-tuple containing:
        - found (bool): True if a path from start to goal was found, else False.
        - events (list[tuple]): Sequence of ('discover', node_id) and ('visit', node_id) events for agent. Node added as discovered when enqueued, then as visited when dequeued & processed.
        - sequence (list[int]): Node IDs in visitation order, appended as each node is popped from the stack and processed. Used for path reconstruction and visualization.
        - path (list[int]): Node IDs forming a path from start to goal inclusive. Empty list if no path was found.
        - parent (dict): Maps each discovered node ID to its parent node ID. Used for path reconstruction and tree visualization.
    """
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]

    parent = {start_id: None}
    queue = deque([start_id])
    events = [('discover', start_id)]
    sequence = []  # visit-only list for path reconstruction

    if start_id == goal_id:
        return True, events, sequence, [start_id], parent
    print("goal: ", goal)
    while queue:
        curr = queue.popleft()
        events.append(('visit', curr))
        sequence.append(curr)
        print('curr: ', curr)

        if curr == goal_id:
            print('goal found')
            print("events: ", events)
            print("sequence: ", sequence)
            return True, events, sequence, _reconstruct_path(parent, start_id, goal_id), parent

        print('adding neighbors: ')
        for neighbor in G.neighbors(curr):
            if neighbor not in parent:
                parent[neighbor] = curr
                events.append(('discover', neighbor))
                queue.append(neighbor)
                print('new neighbor: ', neighbor)


    return False, events, sequence, [], parent

def dfs(G, start, goal, cols):
    """
    Depth-first search on a grid graph from start to goal.

    Parameters
    ----------
    G : networkx.Graph
        Grid graph where nodes are flat integer IDs (r * cols + c) and
        edges connect valid cardinal neighbors.
    start : tuple
        (row, col) coordinates of the start cell.
    goal : tuple
        (row, col) coordinates of the goal cell.
    cols : int
        Number of columns in the grid, used to convert (row, col) to node ID.

    Returns
    -------
    tuple: A 4-tuple containing:
        - found (bool): True if a path from start to goal was found, else False.
        - sequence (list[int]): Node IDs in visitation order, appended as each node is popped from the stack and processed. Used for path reconstruction and visualization.
        - path (list[int]): Node IDs forming a path from start to goal inclusive. Empty list if no path was found.
        - parent (dict): Maps each discovered node ID to its parent node ID. Used for path reconstruction and tree visualization.
    """
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]

    parent = {start_id: None}
    stack = [start_id]
    sequence = []

    visited = {start_id}

    while stack:
        curr = stack.pop()

        if curr in sequence:
            continue

        sequence.append(curr)

        if curr == goal_id:
            return True, sequence, _reconstruct_path(parent, start_id, goal_id), parent

        # DiGraph guarantees neighbor order matches directions in grid_to_graph
        # Push order: left, down, up, right → visit order: right, up, down, left
        for neighbor in G.neighbors(curr):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = curr
                stack.append(neighbor)

    return False, sequence, [], parent
