'''
1. Environment Generation

Source: Repurpose graph creation and visualization code from 2025_Measles_in_SW_Kansas
* Provides graph representation and grid implementation as a starting point
* Rework graph logic: remove weights, add grid size parameter (n×n, default 10×10)

Obstacles
* Randomly generate obstacles and flag nodes with an attribute (e.g. obstacle=True) — do not remove nodes from the graph, as obstructed/unobstructed state is needed to populate the grid
* Update graph edges accordingly after obstacle placement

Start & Goal Nodes
* Randomly select from valid (non-obstacle) nodes
'''

import random
import matplotlib.pyplot as plt
import networkx as nx
import string

def create_grid(size = 10):
    grid = [[True] * size for _ in range(size)]
    return grid

def generate_obstacles(grid: list):
    size = len(grid)
    total_cells = size * size

    # Pick a random percentage between 20% and 30%
    obstacle_percentage = random.uniform(0.20, 0.30)
    num_obstacles = int(total_cells * obstacle_percentage)

    # Generate unique random (row, col) positions
    all_positions = [(r, c) for r in range(size) for c in range(size)]
    obstacle_positions = random.sample(all_positions, num_obstacles)

    # Set chosen cells to False (blocked)
    for row, col in obstacle_positions:
        grid[row][col] = False

    return grid

def get_start_and_goal(grid: list):
    size = len(grid)

    # Collect all valid (True) (row, col) indices
    valid_cells = [(r, c) for r in range(size) for c in range(size) if grid[r][c]]

    if len(valid_cells) < 1:
        raise ValueError("Grid does not have enough open cells to place start and goal.")

    start = random.choice(valid_cells)
    goal = random.choice(valid_cells)

    return start, goal

def get_node_labels(num_vertices):
    labels = {}
    alphabet = string.ascii_uppercase  # 'A' to 'Z'
    base = len(alphabet)

    for i in range(num_vertices):
        label = ''
        index = i
        while True:
            label = alphabet[index % base] + label
            index = index // base - 1
            if index < 0:
                break
        labels[i] = label

    return labels

def grid_to_graph(grid: list):
    rows = len(grid)
    cols = len(grid[0])

    G = nx.Graph()

    # Add every cell as a node, storing its (row, col) position and True/False value
    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c  # flatten (r, c) → single int index
            G.add_node(node_id, row=r, col=c, passable=grid[r][c])

    # Connect each node to its 4-directional neighbors (up/down/left/right)
    # Only open cells (True) can form edges — blocked cells (False) are isolated
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(rows):
        for c in range(cols):
            if not grid[r][c]:  # skip obstacle nodes as edge sources
                continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc]:
                    u = r * cols + c
                    v = nr * cols + nc
                    G.add_edge(u, v)

    return G

def visualize_grid_graph(G, rows, cols, start, goal):
    plt.figure(figsize=(8, 8))

    # Use grid coordinates for layout (mirrors the actual grid visually)
    pos = {node: (data['col'], -data['row']) for node, data in G.nodes(data=True)}

    # Convert (r, c) tuples to flat node IDs
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]

    passable = [n for n, d in G.nodes(data=True) if d['passable'] and n not in (start_id, goal_id)]
    blocked = [n for n, d in G.nodes(data=True) if not d['passable']]

    nx.draw_networkx_nodes(G, pos, nodelist=passable, node_color='lightgreen',
                           node_size=500, edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=blocked, node_color='tomato',
                           node_size=500, edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=[start_id], node_color='gold',
                           node_size=600, edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=[goal_id], node_color='dodgerblue',
                           node_size=600, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray')

    # Reuse get_node_labels from flow_network.py for consistent labelling
    labels = get_node_labels(rows * cols)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')

    # Legend
    plt.scatter([], [], c='lightgreen', label='Open (True)', edgecolors='black', s=100)
    plt.scatter([], [], c='tomato', label='Blocked (False)', edgecolors='black', s=100)
    plt.scatter([], [], c='gold', label='Start', edgecolors='black', s=100)
    plt.scatter([], [], c='dodgerblue', label='Goal', edgecolors='black', s=100)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15), borderaxespad=0)

    plt.title(f"Grid Graph ({rows}×{cols})", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
