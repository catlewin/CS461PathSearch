"""
search_visualization.py
-----------------------
Defines the Visualizer class for animating grid-based search agents.

All rendering is driven from a single Agent instance — grid, events,
parent map, and path are all accessed via agent and agent.grid.
BFS-specific coloring (discovered / frontier nodes in yellow) is gated
on self._is_bfs, derived once at construction via isinstance check.

Classes
-------
Visualizer
    Accepts any Agent subclass and exposes three public animation methods.

Module-level helpers (private)
------------------------------
_hierarchy_pos(tree, root)
    Top-down hierarchical layout for a DiGraph tree.

_build_tree(agent, frame)
    Incrementally builds the search tree DiGraph up to a given frame.

_grid_legend(include_discovered)
    Builds matplotlib legend handles for the grid view.

_tree_legend(include_discovered)
    Builds matplotlib legend handles for the tree view.
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import networkx as nx

from agent import Agent, BFSAgent

_anim = None

# Color index map:  0=obstacle  1=unvisited  2=visited  3=agent  4=path  5=discovered
CMAP     = ListedColormap(['black', 'white', 'lightblue', 'tomato', 'mediumseagreen', 'lightyellow'])
CMAP_DFS = ListedColormap(['black', 'white', 'lightblue', 'tomato', 'mediumseagreen'])


# ---------------------------------------------------------------------------
# Module-level private helpers
# ---------------------------------------------------------------------------

def _hierarchy_pos(tree, root):
    """
    Compute a top-down hierarchical layout for a directed tree.

    Parameters
    ----------
    tree : nx.DiGraph
    root : int

    Returns
    -------
    dict  —  node_id → (x, y)
    """
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


def _build_tree(agent, frame):
    """
    Build a directed tree DiGraph containing all nodes visible at *frame*.

    For BFS the event list contains both 'discover' and 'visit' tuples, so
    all discovered nodes are included. For DFS only visited nodes exist.

    Parameters
    ----------
    agent : Agent
    frame : int

    Returns
    -------
    nx.DiGraph
    """
    tree = nx.DiGraph()
    tree.add_node(agent.grid.start)
    visible = agent.events[:min(frame + 1, len(agent.events))]

    if isinstance(agent, BFSAgent):
        for _event, node_id in visible:
            par = agent.parent.get(node_id)
            if par is not None:
                tree.add_edge(par, node_id)
    else:
        for node_id in [n for _, n in visible]:  # DFS events are also (type, node) now
            par = agent.parent.get(node_id)
            if par is not None:
                tree.add_edge(par, node_id)

    return tree


def _grid_legend(include_discovered=False):
    """
    Build legend handles for the grid view.

    Parameters
    ----------
    include_discovered : bool
        If True, inserts a yellow 'Discovered' patch (BFS only).

    Returns
    -------
    list  —  matplotlib legend handles
    """
    elements = [
        mpatches.Patch(facecolor='black',       label='Obstacle'),
        mpatches.Patch(facecolor='white',        edgecolor='gray', label='Unvisited'),
        mpatches.Patch(facecolor='lightblue',    label='Visited'),
        mpatches.Patch(facecolor='tomato',       label='Agent'),
        mpatches.Patch(facecolor='mediumseagreen', label='Path'),
    ]
    if include_discovered:
        elements.insert(3, mpatches.Patch(facecolor='lightyellow', edgecolor='gray', label='Discovered'))
    elements += [
        plt.scatter([], [], marker='^', color='gold',       s=80, edgecolors='black', label='Start'),
        plt.scatter([], [], marker='X', color='dodgerblue', s=80, edgecolors='black', label='Goal'),
    ]
    return elements


def _tree_legend(include_discovered=False):
    """
    Build legend handles for the tree view.

    Parameters
    ----------
    include_discovered : bool
        If True, inserts a yellow 'Discovered' patch (BFS only).

    Returns
    -------
    list  —  matplotlib legend handles
    """
    elements = [
        mpatches.Patch(facecolor='gold',          edgecolor='black', label='Start'),
        mpatches.Patch(facecolor='dodgerblue',    edgecolor='black', label='Goal'),
        mpatches.Patch(facecolor='lightblue',     edgecolor='black', label='Visited'),
        mpatches.Patch(facecolor='tomato',        edgecolor='black', label='Current'),
        mpatches.Patch(facecolor='mediumseagreen', edgecolor='black', label='Path'),
    ]
    if include_discovered:
        elements.insert(2, mpatches.Patch(facecolor='lightyellow', edgecolor='gray', label='Discovered'))
    return elements


# ---------------------------------------------------------------------------
# Visualizer class
# ---------------------------------------------------------------------------

class Visualizer:
    """
    Animates a completed search agent's traversal and result.

    Accepts any Agent subclass (BFSAgent, DFSAgent, …). BFS-specific
    coloring for frontier/discovered nodes is enabled automatically.

    Parameters
    ----------
    agent : Agent
        A search agent on which .search() has already been called.

    Attributes
    ----------
    agent : Agent
    grid  : Grid        convenience alias for agent.grid
    _is_bfs : bool      True when agent is a BFSAgent instance
    """

    def __init__(self, agent: Agent):
        self.agent   = agent
        self.grid    = agent.grid
        self._is_bfs = isinstance(agent, BFSAgent)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_color_grid(self):
        """Return a (size x size) int array with 0=obstacle, 1=open."""
        g = np.zeros((self.grid.size, self.grid.size), dtype=int)
        for node, data in self.grid.G.nodes(data=True):
            if data['passable']:
                g[data['row']][data['col']] = 1
        return g

    def _node_label(self, node_id):
        """Format a flat node ID as a readable (row,col) string."""
        return f"({node_id // self.grid.size},{node_id % self.grid.size})"

    def _stable_tree_pos(self):
        """
        Compute a stable hierarchical layout from the full parent map.
        Called once per animation so node positions don't shift mid-play.
        """
        full_tree = nx.DiGraph()
        full_tree.add_node(self.grid.start)
        for node_id, par in self.agent.parent.items():
            if par is not None:
                full_tree.add_edge(par, node_id)
        return _hierarchy_pos(full_tree, self.grid.start)

    def _node_colors_for_tree(self, tree, visited_set, discover_set, curr, path_set, done):
        """
        Return an ordered list of colors for every node currently in *tree*.

        Priority (highest first):
            done + in path  → mediumseagreen
            current agent   → tomato
            BFS discovered  → lightyellow
            start           → gold
            goal            → dodgerblue
            otherwise       → lightblue
        """
        colors = []
        for n in tree.nodes():
            if done and n in path_set:
                colors.append('mediumseagreen')
            elif n == curr:
                colors.append('tomato')
            elif self._is_bfs and n in discover_set and n not in visited_set:
                colors.append('lightyellow')
            elif n == self.grid.start:
                colors.append('gold')
            elif n == self.grid.goal:
                colors.append('dodgerblue')
            else:
                colors.append('lightblue')
        return colors

    # ------------------------------------------------------------------
    # Public animation methods
    # ------------------------------------------------------------------

    def show_grid(self):
        """
        Animate the agent moving through the grid cell by cell.

        Visited cells turn light blue; the current agent position is tomato.
        On completion the final path is drawn in green (or red title if none).
        """
        global _anim

        path     = self.agent.reconstruct_path()
        sequence = self.agent.visit_sequence()
        events   = self.agent.events

        color_grid = self._base_color_grid()
        cmap  = CMAP if self._is_bfs else CMAP_DFS
        vmax  = 5    if self._is_bfs else 4

        hold_frames   = 15
        total_frames  = len(events) + hold_frames
        node_state    = {}   # node_id → 'discovered' | 'agent' | 'visited'  (BFS only)

        start_row, start_col = self.grid.G.nodes[self.grid.start]['row'], self.grid.G.nodes[self.grid.start]['col']
        goal_row,  goal_col  = self.grid.G.nodes[self.grid.goal]['row'],  self.grid.G.nodes[self.grid.goal]['col']

        fig, ax = plt.subplots(figsize=(8, 8))
        im      = ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_xticks(np.arange(-0.5, self.grid.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid.size, 1), minor=True)
        ax.grid(which='minor', color='gray', linewidth=0.5)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.scatter(start_col, start_row, marker='^', color='gold',       s=200, zorder=5, edgecolors='black', linewidths=0.8)
        ax.scatter(goal_col,  goal_row,  marker='X', color='dodgerblue', s=200, zorder=5, edgecolors='black', linewidths=0.8)
        title = ax.set_title("Searching...", fontsize=16, fontweight='bold', color='gray')
        ax.legend(handles=_grid_legend(self._is_bfs), loc='upper right',
                  bbox_to_anchor=(1.22, 1), borderaxespad=0, fontsize=8)

        def update(frame):
            if frame < len(events):
                if self._is_bfs:
                    event_type, node_id = events[frame]
                    r, c = self.grid.G.nodes[node_id]['row'], self.grid.G.nodes[node_id]['col']
                    if event_type == 'discover':
                        node_state[node_id] = 'discovered'
                        color_grid[r][c] = 5
                    elif event_type == 'visit':
                        for n, s in list(node_state.items()):
                            if s == 'agent':
                                node_state[n] = 'visited'
                                nr, nc = self.grid.G.nodes[n]['row'], self.grid.G.nodes[n]['col']
                                color_grid[nr][nc] = 2
                        node_state[node_id] = 'agent'
                        color_grid[r][c] = 3
                else:
                    # DFS — only 'visit' events; step through sequence
                    _, node_id = events[frame]
                    if frame > 0:
                        _, prev = events[frame - 1]
                        pr, pc = self.grid.G.nodes[prev]['row'], self.grid.G.nodes[prev]['col']
                        color_grid[pr][pc] = 2
                    r, c = self.grid.G.nodes[node_id]['row'], self.grid.G.nodes[node_id]['col']
                    color_grid[r][c] = 3

                title.set_text(f"Searching — step {frame + 1}/{len(events)}")
                title.set_color('gray')
            else:
                if self.agent.found:
                    for n in path:
                        nr, nc = self.grid.G.nodes[n]['row'], self.grid.G.nodes[n]['col']
                        color_grid[nr][nc] = 4
                    title.set_text("Path Found ✓");  title.set_color('green')
                else:
                    title.set_text("No Path Found ✗"); title.set_color('red')
            im.set_data(color_grid)
            return [im, title]

        _anim = FuncAnimation(fig, update, frames=total_frames, interval=150, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()

    def show_tree(self):
        """
        Animate the search as a growing directed tree.

        Uses a stable hierarchical layout computed from the full parent map
        so nodes don't shift position as the tree grows. For BFS, discovered
        (frontier) nodes appear in yellow before turning light blue on visit.
        """
        global _anim

        path      = self.agent.reconstruct_path()
        path_set  = set(path)
        events    = self.agent.events
        pos       = self._stable_tree_pos()

        hold_frames  = 15
        total_frames = len(events) + hold_frames

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.axis('off')

        def update(frame):
            ax.cla(); ax.axis('off')
            done = frame >= len(events)

            visible       = events[:min(frame + 1, len(events))]
            visited_set   = {n for t, n in visible if t == 'visit'}
            discover_set  = {n for t, n in visible if t == 'discover'} if self._is_bfs else set()

            tree     = _build_tree(self.agent, frame)
            curr     = next((n for t, n in reversed(visible) if t == 'visit'), None) if not done else None
            vis_pos  = {n: pos[n] for n in tree.nodes() if n in pos}

            node_colors  = self._node_colors_for_tree(tree, visited_set, discover_set, curr, path_set, done)
            edge_colors  = [
                'mediumseagreen' if (done and self.agent.found and u in path_set and v in path_set) else 'gray'
                for u, v in tree.edges()
            ]

            nx.draw_networkx_nodes(tree, vis_pos, ax=ax, node_color=node_colors, node_size=600, edgecolors='black', linewidths=0.8)
            nx.draw_networkx_edges(tree, vis_pos, ax=ax, edge_color=edge_colors, arrows=True, arrowsize=12, width=1.5)
            nx.draw_networkx_labels(tree, vis_pos, ax=ax, labels={n: self._node_label(n) for n in tree.nodes()}, font_size=7, font_weight='bold')

            if done:
                ax.set_title("Path Found ✓" if self.agent.found else "No Path Found ✗",
                             fontsize=16, fontweight='bold', color='green' if self.agent.found else 'red')
            else:
                ax.set_title(f"Tree View — step {frame + 1}/{len(events)}", fontsize=16, fontweight='bold', color='gray')

            ax.legend(handles=_tree_legend(self._is_bfs), loc='upper right', fontsize=9)

        _anim = FuncAnimation(fig, update, frames=total_frames, interval=150, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()

    def show_all(self):
        """
        Animate the grid view and tree view side by side in a single window.

        Both subplots are driven by the same FuncAnimation so they stay in sync.
        The left panel shows the grid; the right shows the growing search tree.
        """
        global _anim

        path      = self.agent.reconstruct_path()
        path_set  = set(path)
        events    = self.agent.events
        pos       = self._stable_tree_pos()

        color_grid  = self._base_color_grid()
        cmap        = CMAP if self._is_bfs else CMAP_DFS
        vmax        = 5    if self._is_bfs else 4
        node_state  = {}

        start_row, start_col = self.grid.G.nodes[self.grid.start]['row'], self.grid.G.nodes[self.grid.start]['col']
        goal_row,  goal_col  = self.grid.G.nodes[self.grid.goal]['row'],  self.grid.G.nodes[self.grid.goal]['col']

        hold_frames  = 15
        total_frames = len(events) + hold_frames

        fig, (ax_grid, ax_tree) = plt.subplots(1, 2, figsize=(18, 8))
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        fig.suptitle("Search Visualization", fontsize=18, fontweight='bold')

        im = ax_grid.imshow(color_grid, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
        ax_grid.set_xticks(np.arange(-0.5, self.grid.size, 1), minor=True)
        ax_grid.set_yticks(np.arange(-0.5, self.grid.size, 1), minor=True)
        ax_grid.grid(which='minor', color='gray', linewidth=0.5)
        ax_grid.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax_grid.scatter(start_col, start_row, marker='^', color='gold',       s=200, zorder=5, edgecolors='black', linewidths=0.8)
        ax_grid.scatter(goal_col,  goal_row,  marker='X', color='dodgerblue', s=200, zorder=5, edgecolors='black', linewidths=0.8)
        grid_title = ax_grid.set_title("Grid View — Searching...", fontsize=13, fontweight='bold', color='gray')
        ax_grid.legend(handles=_grid_legend(self._is_bfs), loc='upper right',
                       bbox_to_anchor=(1.22, 1), borderaxespad=0, fontsize=8)
        ax_tree.axis('off')

        def update(frame):
            done = frame >= len(events)

            # ---- Grid panel ----
            if not done:
                if self._is_bfs:
                    event_type, node_id = events[frame]
                    r, c = self.grid.G.nodes[node_id]['row'], self.grid.G.nodes[node_id]['col']
                    if event_type == 'discover':
                        node_state[node_id] = 'discovered'
                        color_grid[r][c] = 5
                    elif event_type == 'visit':
                        for n, s in list(node_state.items()):
                            if s == 'agent':
                                node_state[n] = 'visited'
                                nr, nc = self.grid.G.nodes[n]['row'], self.grid.G.nodes[n]['col']
                                color_grid[nr][nc] = 2
                        node_state[node_id] = 'agent'
                        color_grid[r][c] = 3
                else:
                    _, node_id = events[frame]
                    if frame > 0:
                        _, prev = events[frame - 1]
                        pr, pc = self.grid.G.nodes[prev]['row'], self.grid.G.nodes[prev]['col']
                        color_grid[pr][pc] = 2
                    r, c = self.grid.G.nodes[node_id]['row'], self.grid.G.nodes[node_id]['col']
                    color_grid[r][c] = 3

                grid_title.set_text(f"Grid View — step {frame + 1}/{len(events)}")
                grid_title.set_color('gray')
            else:
                if self.agent.found:
                    for n in path:
                        nr, nc = self.grid.G.nodes[n]['row'], self.grid.G.nodes[n]['col']
                        color_grid[nr][nc] = 4
                    grid_title.set_text("Grid View — Path Found ✓"); grid_title.set_color('green')
                else:
                    grid_title.set_text("Grid View — No Path Found ✗"); grid_title.set_color('red')
            im.set_data(color_grid)

            # ---- Tree panel ----
            ax_tree.cla(); ax_tree.axis('off')

            visible      = events[:min(frame + 1, len(events))]
            visited_set  = {n for t, n in visible if t == 'visit'}
            discover_set = {n for t, n in visible if t == 'discover'} if self._is_bfs else set()

            tree    = _build_tree(self.agent, frame)
            curr    = next((n for t, n in reversed(visible) if t == 'visit'), None) if not done else None
            vis_pos = {n: pos[n] for n in tree.nodes() if n in pos}

            node_colors = self._node_colors_for_tree(tree, visited_set, discover_set, curr, path_set, done)
            edge_colors = [
                'mediumseagreen' if (done and self.agent.found and u in path_set and v in path_set) else 'gray'
                for u, v in tree.edges()
            ]

            nx.draw_networkx_nodes(tree, vis_pos, ax=ax_tree, node_color=node_colors, node_size=500, edgecolors='black', linewidths=0.8)
            nx.draw_networkx_edges(tree, vis_pos, ax=ax_tree, edge_color=edge_colors, arrows=True, arrowsize=12, width=1.5)
            nx.draw_networkx_labels(tree, vis_pos, ax=ax_tree, labels={n: self._node_label(n) for n in tree.nodes()}, font_size=7, font_weight='bold')

            if done:
                ax_tree.set_title("Tree View — Path Found ✓" if self.agent.found else "Tree View — No Path Found ✗",
                                  fontsize=13, fontweight='bold', color='green' if self.agent.found else 'red')
            else:
                ax_tree.set_title(f"Tree View — step {frame + 1}/{len(events)}", fontsize=13, fontweight='bold', color='gray')

            ax_tree.legend(handles=_tree_legend(self._is_bfs), loc='upper right', fontsize=8)

        _anim = FuncAnimation(fig, update, frames=total_frames, interval=250, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()