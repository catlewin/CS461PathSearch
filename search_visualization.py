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

import matplotlib
matplotlib.use('TkAgg')  # otherwise unable to run animation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import networkx as nx

_anim = None

# Color indices
# 0=obstacle, 1=unvisited, 2=visited, 3=agent, 4=path, 5=discovered
CMAP      = ListedColormap(['black', 'white', 'lightblue', 'tomato', 'mediumseagreen', 'lightyellow'])
CMAP_DFS  = ListedColormap(['black', 'white', 'lightblue', 'tomato', 'mediumseagreen'])  # no discovered state


def _hierarchy_pos(tree, root):
    depths = {root: 0}
    for u, v in nx.bfs_edges(tree, root):
        depths[v] = depths[u] + 1
    levels = {}
    for node, depth in depths.items():
        levels.setdefault(depth, []).append(node)
    pos = {}
    for depth, nodes in levels.items():
        for i, node in enumerate(nodes):
            pos[node] = ((i - (len(nodes) - 1) / 2), -depth)
    return pos


def _build_tree(sequence_or_events, parent, start_id, frame, is_bfs=False):
    """Build DiGraph incrementally up to frame."""
    tree = nx.DiGraph()
    tree.add_node(start_id)
    if is_bfs:
        visible = sequence_or_events[:min(frame + 1, len(sequence_or_events))]
        for event_type, node_id in visible:
            par = parent.get(node_id)
            if par is not None:
                tree.add_edge(par, node_id)
    else:
        visible = sequence_or_events[:min(frame + 1, len(sequence_or_events))]
        for node_id in visible[1:]:
            par = parent.get(node_id)
            if par is not None:
                tree.add_edge(par, node_id)
    return tree


def _grid_legend(include_discovered=False):
    elements = [
        mpatches.Patch(facecolor='black',          label='Obstacle'),
        mpatches.Patch(facecolor='white',          edgecolor='gray', label='Unvisited'),
        mpatches.Patch(facecolor='lightblue',      label='Visited'),
        mpatches.Patch(facecolor='tomato',         label='Agent'),
        mpatches.Patch(facecolor='mediumseagreen', label='Path'),
    ]
    if include_discovered:
        elements.insert(3, mpatches.Patch(facecolor='lightyellow', edgecolor='gray', label='Discovered'))
    elements += [
        plt.scatter([], [], marker='^', color='gold',       s=80, edgecolors='black', label='Start'),
        plt.scatter([], [], marker='X', color='dodgerblue', s=80, edgecolors='black', label='Goal'),
    ]
    return elements


def _tree_legend():
    return [
        mpatches.Patch(facecolor='gold',           edgecolor='black', label='Start'),
        mpatches.Patch(facecolor='dodgerblue',     edgecolor='black', label='Goal'),
        mpatches.Patch(facecolor='lightyellow',    edgecolor='gray',  label='Discovered'),
        mpatches.Patch(facecolor='lightblue',      edgecolor='black', label='Visited'),
        mpatches.Patch(facecolor='tomato',         edgecolor='black', label='Current'),
        mpatches.Patch(facecolor='mediumseagreen', edgecolor='black', label='Path'),
    ]


# ---------------------------------------------------------------------------
# Grid only
# ---------------------------------------------------------------------------
def visualize_grid_path(found, sequence, path, G, start, goal, cols):
    global _anim
    rows = max(d['row'] for _, d in G.nodes(data=True)) + 1

    base_grid = np.zeros((rows, cols), dtype=int)
    for node, data in G.nodes(data=True):
        if data['passable']:
            base_grid[data['row']][data['col']] = 1
    color_grid   = base_grid.copy()
    hold_frames  = 15
    total_frames = len(sequence) + hold_frames

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(color_grid, cmap=CMAP_DFS, vmin=0, vmax=4, interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.scatter(start[1], start[0], marker='^', color='gold', s=200, zorder=5, edgecolors='black', linewidths=0.8)
    ax.scatter(goal[1],  goal[0],  marker='X', color='dodgerblue', s=200, zorder=5, edgecolors='black', linewidths=0.8)
    title = ax.set_title("Searching...", fontsize=16, fontweight='bold', color='gray')
    ax.legend(handles=_grid_legend(), loc='upper right', bbox_to_anchor=(1.22, 1), borderaxespad=0, fontsize=8)

    def update(frame):
        if frame < len(sequence):
            if frame > 0:
                pr, pc = sequence[frame-1] // cols, sequence[frame-1] % cols
                color_grid[pr][pc] = 2
            cr, cc = sequence[frame] // cols, sequence[frame] % cols
            color_grid[cr][cc] = 3
        else:
            if found:
                for n in path:
                    color_grid[n // cols][n % cols] = 4
                title.set_text("Path Found ✓"); title.set_color('green')
            else:
                title.set_text("No Path Found ✗"); title.set_color('red')
        im.set_data(color_grid)
        return [im, title]

    _anim = FuncAnimation(fig, update, frames=total_frames, interval=150, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Tree only
# ---------------------------------------------------------------------------
def visualize_tree_search(found, sequence, path, parent, G, start, goal, cols):
    global _anim
    start_id = start[0] * cols + start[1]
    goal_id  = goal[0]  * cols + goal[1]
    path_set = set(path)

    def label(n): return f"({n // cols},{n % cols})"

    full_tree = nx.DiGraph()
    full_tree.add_node(start_id)
    for node_id in sequence[1:]:
        par = parent.get(node_id)
        if par is not None:
            full_tree.add_edge(par, node_id)
    pos = _hierarchy_pos(full_tree, start_id)

    hold_frames  = 15
    total_frames = len(sequence) + hold_frames
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis('off')

    def update(frame):
        ax.cla(); ax.axis('off')
        visible  = sequence[:min(frame + 1, len(sequence))]
        tree     = _build_tree(visible, parent, start_id)
        curr     = visible[-1] if frame < len(sequence) else None
        vis_pos  = {n: pos[n] for n in tree.nodes() if n in pos}

        node_colors = []
        for n in tree.nodes():
            if frame >= len(sequence) and found and n in path_set: node_colors.append('mediumseagreen')
            elif n == curr:     node_colors.append('tomato')
            elif n == start_id: node_colors.append('gold')
            elif n == goal_id:  node_colors.append('dodgerblue')
            else:               node_colors.append('lightblue')

        edge_colors = ['mediumseagreen' if (frame >= len(sequence) and found and u in path_set and v in path_set)
                       else 'gray' for u, v in tree.edges()]

        nx.draw_networkx_nodes(tree, vis_pos, ax=ax, node_color=node_colors, node_size=600, edgecolors='black', linewidths=0.8)
        nx.draw_networkx_edges(tree, vis_pos, ax=ax, edge_color=edge_colors, arrows=True, arrowsize=12, width=1.5)
        nx.draw_networkx_labels(tree, vis_pos, ax=ax, labels={n: label(n) for n in tree.nodes()}, font_size=7, font_weight='bold')

        if frame >= len(sequence):
            ax.set_title("Path Found ✓" if found else "No Path Found ✗",
                         fontsize=16, fontweight='bold', color='green' if found else 'red')
        else:
            ax.set_title(f"Tree View — step {frame+1}/{len(sequence)}", fontsize=16, fontweight='bold', color='gray')
        ax.legend(handles=_tree_legend(), loc='upper right', fontsize=9)

    _anim = FuncAnimation(fig, update, frames=total_frames, interval=150, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Side by side (BFS-aware with discovered state)
# ---------------------------------------------------------------------------
def visualize_side_by_side(found, events_or_sequence, sequence, path, parent, G, start, goal, cols, is_bfs=False):
    global _anim
    start_id = start[0] * cols + start[1]
    goal_id  = goal[0]  * cols + goal[1]
    path_set = set(path)
    rows     = max(d['row'] for _, d in G.nodes(data=True)) + 1

    def label(n): return f"({n // cols},{n % cols})"

    # Grid setup
    base_grid = np.zeros((rows, cols), dtype=int)
    for node, data in G.nodes(data=True):
        if data['passable']:
            base_grid[data['row']][data['col']] = 1
    color_grid = base_grid.copy()
    cmap       = CMAP if is_bfs else CMAP_DFS
    vmax       = 5 if is_bfs else 4

    # Tree layout — built from visit-only sequence for stable pos
    full_tree = nx.DiGraph()
    full_tree.add_node(start_id)
    for node_id in sequence[1:]:
        par = parent.get(node_id)
        if par is not None:
            full_tree.add_edge(par, node_id)
    pos = _hierarchy_pos(full_tree, start_id)

    events       = events_or_sequence  # for BFS: list of (type, node); for DFS: list of node ints
    hold_frames  = 15
    total_frames = len(events) + hold_frames

    # Track per-node state for BFS coloring
    node_state = {}  # node_id -> 'discovered' | 'visited'

    fig, (ax_grid, ax_tree) = plt.subplots(1, 2, figsize=(18, 8))
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    fig.suptitle("Search Visualization", fontsize=18, fontweight='bold')

    im = ax_grid.imshow(color_grid, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    ax_grid.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax_grid.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax_grid.grid(which='minor', color='gray', linewidth=0.5)
    ax_grid.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_grid.scatter(start[1], start[0], marker='^', color='gold', s=200, zorder=5, edgecolors='black', linewidths=0.8)
    ax_grid.scatter(goal[1],  goal[0],  marker='X', color='dodgerblue', s=200, zorder=5, edgecolors='black', linewidths=0.8)
    grid_title = ax_grid.set_title("Grid View — Searching...", fontsize=13, fontweight='bold', color='gray')
    ax_grid.legend(handles=_grid_legend(include_discovered=is_bfs), loc='upper right',
                   bbox_to_anchor=(1.22, 1), borderaxespad=0, fontsize=8)
    ax_tree.axis('off')

    def update(frame):
        # ---- Grid update ----
        if frame < len(events):
            if is_bfs:
                event_type, node_id = events[frame]
                r, c = node_id // cols, node_id % cols
                if event_type == 'discover':
                    node_state[node_id] = 'discovered'
                    color_grid[r][c] = 5          # light yellow
                elif event_type == 'visit':
                    # clear previous agent marker
                    prev_agent = [n for n, s in node_state.items() if s == 'agent']
                    for n in prev_agent:
                        node_state[n] = 'visited'
                        pr, pc = n // cols, n % cols
                        color_grid[pr][pc] = 2
                    node_state[node_id] = 'agent'
                    color_grid[r][c] = 3          # tomato (agent)
            else:
                # DFS — plain sequence
                if frame > 0:
                    prev = events[frame - 1]
                    color_grid[prev // cols][prev % cols] = 2
                curr = events[frame]
                color_grid[curr // cols][curr % cols] = 3

            grid_title.set_text(f"Grid View — step {frame+1}/{len(events)}")
            grid_title.set_color('gray')
        else:
            if found:
                for n in path:
                    color_grid[n // cols][n % cols] = 4
                grid_title.set_text("Grid View — Path Found ✓"); grid_title.set_color('green')
            else:
                grid_title.set_text("Grid View — No Path Found ✗"); grid_title.set_color('red')
        im.set_data(color_grid)

        # ---- Tree update ----
        ax_tree.cla(); ax_tree.axis('off')

        # Build tree from visits only up to this frame
        if is_bfs:
            visited_so_far  = [n for t, n in events[:min(frame+1, len(events))] if t == 'visit']
            discover_so_far = {n for t, n in events[:min(frame+1, len(events))] if t == 'discover'}
        else:
            visited_so_far  = events[:min(frame+1, len(events))]
            discover_so_far = set()

        tree = nx.DiGraph()
        tree.add_node(start_id)
        all_shown = set(visited_so_far) | discover_so_far
        for node_id in all_shown:
            if node_id == start_id: continue
            par = parent.get(node_id)
            if par is not None:
                tree.add_edge(par, node_id)

        curr     = visited_so_far[-1] if visited_so_far and frame < len(events) else None
        vis_pos  = {n: pos[n] for n in tree.nodes() if n in pos}

        node_colors = []
        for n in tree.nodes():
            if frame >= len(events) and found and n in path_set:
                node_colors.append('mediumseagreen')
            elif n == curr:
                node_colors.append('tomato')
            elif is_bfs and n in discover_so_far and n not in visited_so_far:
                node_colors.append('lightyellow')
            elif n == start_id:
                node_colors.append('gold')
            elif n == goal_id:
                node_colors.append('dodgerblue')
            else:
                node_colors.append('lightblue')

        edge_colors = ['mediumseagreen' if (frame >= len(events) and found and u in path_set and v in path_set)
                       else 'gray' for u, v in tree.edges()]

        nx.draw_networkx_nodes(tree, vis_pos, ax=ax_tree, node_color=node_colors, node_size=500, edgecolors='black', linewidths=0.8)
        nx.draw_networkx_edges(tree, vis_pos, ax=ax_tree, edge_color=edge_colors, arrows=True, arrowsize=12, width=1.5)
        nx.draw_networkx_labels(tree, vis_pos, ax=ax_tree, labels={n: label(n) for n in tree.nodes()}, font_size=7, font_weight='bold')

        if frame >= len(events):
            ax_tree.set_title("Tree View — Path Found ✓" if found else "Tree View — No Path Found ✗",
                              fontsize=13, fontweight='bold', color='green' if found else 'red')
        else:
            ax_tree.set_title(f"Tree View — step {frame+1}/{len(events)}", fontsize=13, fontweight='bold', color='gray')

        legend = _tree_legend() if is_bfs else _tree_legend()[2:]  # drop discovered entry for DFS
        ax_tree.legend(handles=legend, loc='upper right', fontsize=8)

    _anim = FuncAnimation(fig, update, frames=total_frames, interval=150, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()