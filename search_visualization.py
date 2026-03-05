'''
3. Search Visualization

Approach: Run full search first, then replay stored history frame-by-frame using matplotlib.animation.FuncAnimation
* Reference: matplotlib animation docs
* Grid and tree displayed side-by-side in the same window
* Use GenAI support for: representing obstacles, start/goal icons, and agent position in grid and graph

Grid View
Element	        Representation
Unvisited cell	White
Visited cell	Light blue
Obstructed cell	Black
Start/Goal node	Icon (triangle, x)
Agent position	Circle
Final path      Green

Tree View
Element	        Representation
Root	        Start node
Each step	    New child node connected to previous
Current node	Marked with X
Final path      Green
'''

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5 installed

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import networkx as nx

anim = None

def visualize_grid_path(found, sequence, path, G, start, goal, cols):
    global anim
    # --- Derive grid dimensions ---
    rows = max(d['row'] for _, d in G.nodes(data=True)) + 1

    # --- Color indices ---
    # 0 = obstacle, 1 = unvisited open, 2 = visited, 3 = agent, 4 = path
    cmap = ListedColormap(['black', 'white', 'lightblue', 'tomato', 'mediumseagreen'])

    # --- Build initial color grid ---
    base_grid = np.zeros((rows, cols), dtype=int)
    for node, data in G.nodes(data=True):
        if data['passable']:
            base_grid[data['row']][data['col']] = 1

    # Working copy updated each frame
    color_grid = base_grid.copy()

    path_set = set(path)
    hold_frames = 15  # extra frames at end to display final state
    total_frames = len(sequence) + hold_frames

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')

    # --- Gridlines ---
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    # --- Static start / goal markers (always on top) ---
    ax.scatter(start[1], start[0], marker='^', color='gold',
               s=200, zorder=5, edgecolors='black', linewidths=0.8)
    ax.scatter(goal[1], goal[0], marker='X', color='dodgerblue',
               s=200, zorder=5, edgecolors='black', linewidths=0.8)

    # --- Status title (updated on completion) ---
    title = ax.set_title("Searching...", fontsize=16, fontweight='bold', color='gray')

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor='black',           label='Obstacle'),
        mpatches.Patch(facecolor='white',           edgecolor='gray', label='Unvisited'),
        mpatches.Patch(facecolor='lightblue',       label='Visited'),
        mpatches.Patch(facecolor='tomato',          label='Agent'),
        mpatches.Patch(facecolor='mediumseagreen',  label='Path'),
        plt.scatter([], [], marker='^', color='gold',       s=100, edgecolors='black', label='Start'),
        plt.scatter([], [], marker='X', color='dodgerblue', s=100, edgecolors='black', label='Goal'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.22, 1), borderaxespad=0, fontsize=9)

    # --- Animation update function ---
    def update(frame):
        if frame < len(sequence):
            # Mark previous agent position as visited
            if frame > 0:
                prev_id = sequence[frame - 1]
                pr, pc = prev_id // cols, prev_id % cols
                color_grid[pr][pc] = 2

            # Mark current agent position
            curr_id = sequence[frame]
            cr, cc = curr_id // cols, curr_id % cols
            color_grid[cr][cc] = 3

        else:
            # Hold frames — overlay final path or show no-path state
            if found:
                for node_id in path:
                    r, c = node_id // cols, node_id % cols
                    color_grid[r][c] = 4
                title.set_text("Path Found ✓")
                title.set_color('green')
            else:
                title.set_text("No Path Found ✗")
                title.set_color('red')

        im.set_data(color_grid)
        return [im, title]

    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=300, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()

def hierarchy_pos(tree, root):
    # Assign depth levels via BFS
    depths = {root: 0}
    for u, v in nx.bfs_edges(tree, root):
        depths[v] = depths[u] + 1

    # Group nodes by depth
    levels = {}
    for node, depth in depths.items():
        levels.setdefault(depth, []).append(node)

    # Assign (x, y) — y flipped so root is at top
    pos = {}
    for depth, nodes in levels.items():
        for i, node in enumerate(nodes):
            x = (i - (len(nodes) - 1) / 2)  # center each level
            y = -depth                        # negative so root is highest
            pos[node] = (x, y)

    return pos

def visualize_tree_search(found, sequence, path, parent, G, start, goal, cols):
    global _anim

    start_id = start[0] * cols + start[1]
    goal_id  = goal[0]  * cols + goal[1]
    path_set = set(path)

    def label(node_id):
        r, c = node_id // cols, node_id % cols
        return f"({r},{c})"

    # Build full tree upfront for a stable layout across all frames
    full_tree = nx.DiGraph()
    full_tree.add_node(start_id)
    for node_id in sequence[1:]:
        par = parent.get(node_id)
        if par is not None:
            full_tree.add_edge(par, node_id)

    pos = hierarchy_pos(full_tree, start_id)

    hold_frames  = 15
    total_frames = len(sequence) + hold_frames

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis('off')

    def update(frame):
        ax.cla()
        ax.axis('off')

        # Incrementally build tree up to current frame
        visible = sequence[:min(frame + 1, len(sequence))]
        tree = nx.DiGraph()
        tree.add_node(start_id)
        for node_id in visible[1:]:
            par = parent.get(node_id)
            if par is not None:
                tree.add_edge(par, node_id)

        curr = visible[-1] if frame < len(sequence) else None

        # Node colors
        node_colors = []
        for n in tree.nodes():
            if frame >= len(sequence) and found and n in path_set:
                node_colors.append('mediumseagreen')
            elif n == curr:
                node_colors.append('tomato')
            elif n == start_id:
                node_colors.append('gold')
            elif n == goal_id:
                node_colors.append('dodgerblue')
            else:
                node_colors.append('lightblue')

        # Edge colors
        edge_colors = []
        for u, v in tree.edges():
            if frame >= len(sequence) and found and u in path_set and v in path_set:
                edge_colors.append('mediumseagreen')
            else:
                edge_colors.append('gray')

        visible_pos = {n: pos[n] for n in tree.nodes() if n in pos}

        nx.draw_networkx_nodes(tree, visible_pos, ax=ax,
                               node_color=node_colors, node_size=600,
                               edgecolors='black', linewidths=0.8)
        nx.draw_networkx_edges(tree, visible_pos, ax=ax,
                               edge_color=edge_colors, arrows=True,
                               arrowsize=12, width=1.5)
        nx.draw_networkx_labels(tree, visible_pos, ax=ax,
                                labels={n: label(n) for n in tree.nodes()},
                                font_size=7, font_weight='bold')

        # Title
        if frame >= len(sequence):
            if found:
                ax.set_title("Path Found ✓", fontsize=16, fontweight='bold', color='green')
            else:
                ax.set_title("No Path Found ✗", fontsize=16, fontweight='bold', color='red')
        else:
            ax.set_title(f"Searching... step {frame + 1}/{len(sequence)}",
                         fontsize=16, fontweight='bold', color='gray')

        legend_elements = [
            mpatches.Patch(facecolor='gold',           edgecolor='black', label='Start'),
            mpatches.Patch(facecolor='dodgerblue',     edgecolor='black', label='Goal'),
            mpatches.Patch(facecolor='lightblue',      edgecolor='black', label='Visited'),
            mpatches.Patch(facecolor='tomato',         edgecolor='black', label='Current'),
            mpatches.Patch(facecolor='mediumseagreen', edgecolor='black', label='Path'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    _anim = FuncAnimation(fig, update, frames=total_frames,
                          interval=150, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

def visualize_side_by_side(found, sequence, path, parent, G, start, goal, cols):
    global _anim

    start_id = start[0] * cols + start[1]
    goal_id  = goal[0]  * cols + goal[1]
    path_set = set(path)
    rows     = max(d['row'] for _, d in G.nodes(data=True)) + 1

    def label(node_id):
        r, c = node_id // cols, node_id % cols
        return f"({r},{c})"

    # --- Grid setup ---
    cmap = ListedColormap(['black', 'white', 'lightblue', 'tomato', 'mediumseagreen'])
    base_grid = np.zeros((rows, cols), dtype=int)
    for node, data in G.nodes(data=True):
        if data['passable']:
            base_grid[data['row']][data['col']] = 1
    color_grid = base_grid.copy()

    # --- Tree setup: build full tree for stable layout ---
    def hierarchy_pos(tree, root):
        depths = {root: 0}
        for u, v in nx.bfs_edges(tree, root):
            depths[v] = depths[u] + 1
        levels = {}
        for node, depth in depths.items():
            levels.setdefault(depth, []).append(node)
        pos = {}
        for depth, nodes in levels.items():
            for i, node in enumerate(nodes):
                x = (i - (len(nodes) - 1) / 2)
                y = -depth
                pos[node] = (x, y)
        return pos

    full_tree = nx.DiGraph()
    full_tree.add_node(start_id)
    for node_id in sequence[1:]:
        par = parent.get(node_id)
        if par is not None:
            full_tree.add_edge(par, node_id)
    pos = hierarchy_pos(full_tree, start_id)

    hold_frames  = 15
    total_frames = len(sequence) + hold_frames

    # --- Figure with two subplots ---
    fig, (ax_grid, ax_tree) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Search Visualization", fontsize=18, fontweight='bold')

    # Grid subplot init
    im = ax_grid.imshow(color_grid, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
    ax_grid.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax_grid.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax_grid.grid(which='minor', color='gray', linewidth=0.5)
    ax_grid.tick_params(which='both', bottom=False, left=False,
                        labelbottom=False, labelleft=False)
    ax_grid.scatter(start[1], start[0], marker='^', color='gold',
                    s=200, zorder=5, edgecolors='black', linewidths=0.8)
    ax_grid.scatter(goal[1], goal[0], marker='X', color='dodgerblue',
                    s=200, zorder=5, edgecolors='black', linewidths=0.8)
    grid_title = ax_grid.set_title("Grid View — Searching...",
                                   fontsize=13, fontweight='bold', color='gray')

    grid_legend = [
        mpatches.Patch(facecolor='black',          label='Obstacle'),
        mpatches.Patch(facecolor='white',          edgecolor='gray', label='Unvisited'),
        mpatches.Patch(facecolor='lightblue',      label='Visited'),
        mpatches.Patch(facecolor='tomato',         label='Agent'),
        mpatches.Patch(facecolor='mediumseagreen', label='Path'),
        plt.scatter([], [], marker='^', color='gold',       s=80, edgecolors='black', label='Start'),
        plt.scatter([], [], marker='X', color='dodgerblue', s=80, edgecolors='black', label='Goal'),
    ]
    ax_grid.legend(handles=grid_legend, loc='upper right',
                   bbox_to_anchor=(1.22, 1), borderaxespad=0, fontsize=8)

    # Tree subplot init
    ax_tree.axis('off')

    def update(frame):
        # ---- Grid update ----
        if frame < len(sequence):
            if frame > 0:
                prev_id = sequence[frame - 1]
                pr, pc  = prev_id // cols, prev_id % cols
                color_grid[pr][pc] = 2
            curr_id = sequence[frame]
            cr, cc  = curr_id // cols, curr_id % cols
            color_grid[cr][cc] = 3
            grid_title.set_text(f"Grid View — step {frame + 1}/{len(sequence)}")
            grid_title.set_color('gray')
        else:
            if found:
                for node_id in path:
                    r, c = node_id // cols, node_id % cols
                    color_grid[r][c] = 4
                grid_title.set_text("Grid View — Path Found ✓")
                grid_title.set_color('green')
            else:
                grid_title.set_text("Grid View — No Path Found ✗")
                grid_title.set_color('red')
        im.set_data(color_grid)

        # ---- Tree update ----
        ax_tree.cla()
        ax_tree.axis('off')

        visible = sequence[:min(frame + 1, len(sequence))]
        tree = nx.DiGraph()
        tree.add_node(start_id)
        for node_id in visible[1:]:
            par = parent.get(node_id)
            if par is not None:
                tree.add_edge(par, node_id)

        curr = visible[-1] if frame < len(sequence) else None

        node_colors = []
        for n in tree.nodes():
            if frame >= len(sequence) and found and n in path_set:
                node_colors.append('mediumseagreen')
            elif n == curr:
                node_colors.append('tomato')
            elif n == start_id:
                node_colors.append('gold')
            elif n == goal_id:
                node_colors.append('dodgerblue')
            else:
                node_colors.append('lightblue')

        edge_colors = []
        for u, v in tree.edges():
            if frame >= len(sequence) and found and u in path_set and v in path_set:
                edge_colors.append('mediumseagreen')
            else:
                edge_colors.append('gray')

        visible_pos = {n: pos[n] for n in tree.nodes() if n in pos}

        nx.draw_networkx_nodes(tree, visible_pos, ax=ax_tree,
                               node_color=node_colors, node_size=500,
                               edgecolors='black', linewidths=0.8)
        nx.draw_networkx_edges(tree, visible_pos, ax=ax_tree,
                               edge_color=edge_colors, arrows=True,
                               arrowsize=12, width=1.5)
        nx.draw_networkx_labels(tree, visible_pos, ax=ax_tree,
                                labels={n: label(n) for n in tree.nodes()},
                                font_size=7, font_weight='bold')

        if curr is not None and curr in visible_pos:
            x, y = visible_pos[curr]
            ax_tree.scatter(x, y, marker='x', color='black', s=200, zorder=5, linewidths=2)

        if frame >= len(sequence):
            if found:
                ax_tree.set_title("Tree View — Path Found ✓",
                                  fontsize=13, fontweight='bold', color='green')
            else:
                ax_tree.set_title("Tree View — No Path Found ✗",
                                  fontsize=13, fontweight='bold', color='red')
        else:
            ax_tree.set_title(f"Tree View — step {frame + 1}/{len(sequence)}",
                              fontsize=13, fontweight='bold', color='gray')

        tree_legend = [
            mpatches.Patch(facecolor='gold',           edgecolor='black', label='Start'),
            mpatches.Patch(facecolor='dodgerblue',     edgecolor='black', label='Goal'),
            mpatches.Patch(facecolor='lightblue',      edgecolor='black', label='Visited'),
            mpatches.Patch(facecolor='tomato',         edgecolor='black', label='Current'),
            mpatches.Patch(facecolor='mediumseagreen', edgecolor='black', label='Path'),
        ]
        ax_tree.legend(handles=tree_legend, loc='upper right', fontsize=8)

    _anim = FuncAnimation(fig, update, frames=total_frames,
                          interval=150, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

from environment_generation import create_grid, generate_obstacles, get_start_and_goal, grid_to_graph
from agent import bfs, dfs

grid = create_grid()
grid = generate_obstacles(grid)
start, goal = get_start_and_goal(grid)
G = grid_to_graph(grid)
cols = len(grid[0])

found, sequence, path, parent = dfs(G, start, goal, cols)
visualize_side_by_side(found, sequence, path, parent, G, start, goal, cols)