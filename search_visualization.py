"""
search_visualization.py
-----------------------
Defines the Visualizer class for animating grid-based search agents.

All rendering is driven from a single Agent instance — grid, events,
parent map, and path are all accessed via agent and agent.grid.
BFS-specific coloring (discovered / frontier nodes in yellow) is gated
on self._shows_discovered, derived once at construction via isinstance check.

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

from agent import Agent

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


def _build_tree(agent, frame, parent_override=None):
    """
    Build a directed tree DiGraph containing all nodes visible at *frame*.

    Skips 'new_iteration', 'push', and 'parent_snapshot' events.
    Only 'visit' and 'discover' events contribute nodes/edges.

    Parameters
    ----------
    agent : Agent
    frame : int
    parent_override : dict or None
        If provided, used instead of agent.parent for edge construction.
        Used by IDDFS to supply the iteration-correct parent map.

    Returns
    -------
    nx.DiGraph
    """
    parent = parent_override if parent_override is not None else agent.parent
    tree   = nx.DiGraph()
    tree.add_node(agent.grid.start)
    visible = agent.events[:min(frame + 1, len(agent.events))]

    for event_type, payload in visible:
        if event_type not in ('visit', 'discover'):
            continue
        node_id = payload
        par = parent.get(node_id)
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
        self._shows_discovered = agent.shows_discovered
        # True when the environment is a CityGraph or RandomGraph (not a Grid)
        self._is_graph = hasattr(self.grid, 'pos') and hasattr(self.grid, 'labels')

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
        """Return a readable label for a node in any environment type."""
        if self._is_graph:
            return self.grid.labels.get(node_id, str(node_id))
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
            elif self._shows_discovered and n in discover_set and n not in visited_set:
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
        cmap  = CMAP if self._shows_discovered else CMAP_DFS
        vmax  = 5    if self._shows_discovered else 4

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
        ax.legend(handles=_grid_legend(self._shows_discovered), loc='upper right',
                  bbox_to_anchor=(1.22, 1), borderaxespad=0, fontsize=8)

        def update(frame):
            if frame < len(events):
                event_type, payload = events[frame]

                if event_type == 'new_iteration':
                    # Reset grid colors to base and show depth limit in title
                    color_grid[:] = self._base_color_grid()
                    node_state.clear()
                    title.set_text(f"Iteration — depth limit {payload}")
                    title.set_color('steelblue')

                elif self._shows_discovered:
                    node_id = payload
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
                        title.set_text(f"Searching — step {frame + 1}/{len(events)}")
                        title.set_color('gray')

                else:
                    # DFS-like — only 'visit' events carry node IDs
                    if event_type == 'visit':
                        node_id = payload
                        # Find previous visit event for agent trail
                        prev_visit = next(
                            (n for t, n in reversed(events[:frame]) if t == 'visit'),
                            None
                        )
                        if prev_visit is not None:
                            pr, pc = self.grid.G.nodes[prev_visit]['row'], self.grid.G.nodes[prev_visit]['col']
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
            discover_set  = {n for t, n in visible if t == 'discover'} if self._shows_discovered else set()

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

            ax.legend(handles=_tree_legend(self._shows_discovered), loc='upper right', fontsize=9)

        _anim = FuncAnimation(fig, update, frames=total_frames, interval=150, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()

    def show_all(self, metrics=None):
        """
        Animate the search.  Routes to graph or grid implementation
        depending on the environment type.

        Parameters
        ----------
        metrics : MetricsResult, optional
            If provided (from run_with_metrics()), a summary panel is shown
            when the animation completes.
        """
        if self._is_graph:
            self._show_all_graph(metrics=metrics)
        else:
            self._show_all_grid(metrics=metrics)

    def _show_all_grid(self, metrics=None):
        """
        Animate grid, search tree, and open-list widget side by side.

        Left panel   : grid with color-coded cell states
        Middle panel : growing search tree
        Right panel  : live open list (queue / stack / priority queue)

        Bottom strip : Play/Pause, Step, Restart, Speed slider
        Info box     : per-node hover details below the grid
        """
        global _anim

        from agent import BFSAgent, DFSAgent, IDDFSAgent, GreedyAgent, AStarAgent

        path      = self.agent.reconstruct_path()
        path_set  = set(path)
        events    = self.agent.events
        pos       = self._stable_tree_pos()

        base_grid   = self._base_color_grid()
        color_grid  = base_grid.copy()
        cmap        = CMAP if self._shows_discovered else CMAP_DFS
        vmax        = 5    if self._shows_discovered else 4
        node_state  = {}

        start_row, start_col = (self.grid.G.nodes[self.grid.start]['row'],
                                self.grid.G.nodes[self.grid.start]['col'])
        goal_row, goal_col   = (self.grid.G.nodes[self.grid.goal]['row'],
                                self.grid.G.nodes[self.grid.goal]['col'])

        hold_frames  = 15
        total_frames = len(events) + hold_frames
        state        = {'frame': 0, 'playing': True}

        self.agent.visit_sequence()
        g_cost = getattr(self.agent, 'g_cost', None)
        h_cost = getattr(self.agent, 'h_cost', None)

        # hover_parent mirrors the current iteration's parent map for hover
        # lookups. Updated each frame from live_parent so it's always in sync
        # with what's visible on the grid, regardless of which iteration we're in.
        hover_parent = {}

        # ------------------------------------------------------------------
        # Open-list mirror setup
        # open_list  : ordered list of node_ids currently in the open set
        # _ol_label  : function node_id → display string
        # _ol_title  : static header for the queue panel
        # _ol_add / _ol_remove : how to insert / remove from the mirror
        # ------------------------------------------------------------------
        open_list = []   # mutable, shared with update() closure

        if isinstance(self.agent, BFSAgent):
            _ol_title  = 'Queue  (front → back)'
            def _ol_add(n):    open_list.append(n)
            def _ol_remove(n):
                if n in open_list: open_list.remove(n)
            def _ol_label(n):  return self._node_label(n)

        elif isinstance(self.agent, (DFSAgent, IDDFSAgent)):
            _ol_title  = 'Stack  (top → bottom)'
            def _ol_add(n):    open_list.append(n)
            def _ol_remove(n):
                if n in open_list: open_list.remove(n)
            def _ol_label(n):  return self._node_label(n)

        elif isinstance(self.agent, GreedyAgent):
            _ol_title  = 'Open List  ↑ h(n)'
            def _ol_add(n):
                if n in open_list: open_list.remove(n)
                open_list.append(n)
                open_list.sort(key=lambda x: h_cost.get(x, float('inf')))
            def _ol_remove(n):
                if n in open_list: open_list.remove(n)
            def _ol_label(n):
                h = h_cost.get(n, '?')
                return f'{self._node_label(n)}  h={h}'

        else:  # AStarAgent
            _ol_title  = 'Open List  ↑ f(n)'
            def _f(n): return g_cost.get(n, 0) + h_cost.get(n, 0)
            def _ol_add(n):
                if n in open_list: open_list.remove(n)
                open_list.append(n)
                open_list.sort(key=_f)
            def _ol_remove(n):
                if n in open_list: open_list.remove(n)
            def _ol_label(n):
                g = g_cost.get(n, '?')
                h = h_cost.get(n, '?')
                f = (g + h) if isinstance(g, int) and isinstance(h, int) else '?'
                return f'{self._node_label(n)}  {g}+{h}={f}'

        # current_parent tracks the parent map for the active IDDFS iteration.
        # Updated whenever a 'parent_snapshot' event is encountered.
        # For non-IDDFS agents this stays None and _build_tree uses agent.parent.
        iter_parent = [None]   # list so closure can mutate it
        MAX_VISIBLE = 12   # truncate list display beyond this

        # ------------------------------------------------------------------
        # Figure layout — three panels + bottom strip
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(22, 8))
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.20, top=0.93,
                            wspace=0.25)

        # Three main panels using GridSpec: grid=3, tree=5, queue=2 (ratio)
        gs = fig.add_gridspec(1, 3, width_ratios=[4, 4, 2],
                              left=0.03, right=0.97,
                              bottom=0.20, top=0.93, wspace=0.25)
        ax_grid  = fig.add_subplot(gs[0])
        ax_tree  = fig.add_subplot(gs[1])
        ax_queue = fig.add_subplot(gs[2])

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        fig.suptitle("Search Visualization", fontsize=18, fontweight='bold')

        # Info text box below grid
        ax_info = fig.add_axes([0.03, 0.10, 0.30, 0.07])
        ax_info.axis('off')
        info_text = ax_info.text(
            0.01, 0.95, '',
            transform=ax_info.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9)
        )

        # Control buttons
        ax_restart   = fig.add_axes([0.05, 0.03, 0.07, 0.05])
        ax_step      = fig.add_axes([0.13, 0.03, 0.07, 0.05])
        ax_playpause = fig.add_axes([0.21, 0.03, 0.07, 0.05])
        ax_speed     = fig.add_axes([0.42, 0.035, 0.35, 0.03])

        from matplotlib.widgets import Button, Slider
        btn_restart   = Button(ax_restart,   'Restart', color='0.85', hovercolor='0.75')
        btn_step      = Button(ax_step,      'Step',    color='0.85', hovercolor='0.75')
        btn_playpause = Button(ax_playpause, 'Pause',   color='0.85', hovercolor='0.75')
        sld_speed     = Slider(ax_speed, 'ms/frame', 50, 1000,
                               valinit=250, valstep=50, color='steelblue')

        # ------------------------------------------------------------------
        # Grid panel static setup
        # ------------------------------------------------------------------
        im = ax_grid.imshow(color_grid, cmap=cmap, vmin=0, vmax=vmax,
                            interpolation='nearest')
        ax_grid.set_xticks(np.arange(-0.5, self.grid.size, 1), minor=True)
        ax_grid.set_yticks(np.arange(-0.5, self.grid.size, 1), minor=True)
        ax_grid.grid(which='minor', color='gray', linewidth=0.5)
        ax_grid.tick_params(which='both', bottom=False, left=False,
                            labelbottom=False, labelleft=False)
        ax_grid.scatter(start_col, start_row, marker='^', color='gold',
                        s=200, zorder=5, edgecolors='black', linewidths=0.8)
        ax_grid.scatter(goal_col, goal_row, marker='X', color='dodgerblue',
                        s=200, zorder=5, edgecolors='black', linewidths=0.8)
        grid_title = ax_grid.set_title("Grid View — Searching...",
                                       fontsize=13, fontweight='bold', color='gray')
        ax_grid.legend(handles=_grid_legend(self._shows_discovered),
                       loc='upper right', bbox_to_anchor=(1.22, 1),
                       borderaxespad=0, fontsize=8)
        ax_tree.axis('off')

        # ------------------------------------------------------------------
        # Queue panel draw helper — called every frame from update()
        # ------------------------------------------------------------------
        def _draw_queue(next_node):
            """Redraw ax_queue showing current open_list state.

            Colors mirror the main visualization:
                tomato      — next node to be visited (top of queue/stack)
                lightyellow — other frontier nodes
            """
            ax_queue.cla()
            ax_queue.axis('off')
            ax_queue.set_title(_ol_title, fontsize=9, fontweight='bold',
                               color='steelblue', pad=4)

            if not open_list:
                ax_queue.text(0.5, 0.5, '(empty)', ha='center', va='center',
                              fontsize=9, color='gray',
                              transform=ax_queue.transAxes)
                return

            # Stacks show top-first; queues/priority show front-first
            is_stack = isinstance(self.agent, (DFSAgent, IDDFSAgent))
            display  = list(reversed(open_list)) if is_stack else list(open_list)

            truncated = len(display) > MAX_VISIBLE
            visible   = display[:MAX_VISIBLE]

            n_rows = len(visible) + (1 if truncated else 0)
            row_h  = 1.0 / (n_rows + 1)

            for i, node_id in enumerate(visible):
                y       = 1.0 - (i + 1) * row_h
                is_next = (node_id == next_node)
                bg      = 'tomato'      if is_next else 'lightyellow'
                edge    = 'darkred'     if is_next else '#aaa'
                fw      = 'bold'        if is_next else 'normal'
                fc      = 'white'       if is_next else 'black'
                label   = _ol_label(node_id)

                ax_queue.text(
                    0.5, y, label,
                    ha='center', va='center', fontsize=8,
                    fontweight=fw, color=fc,
                    transform=ax_queue.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=bg,
                              edgecolor=edge, linewidth=0.8)
                )

                if i < len(visible) - 1:
                    ax_queue.annotate(
                        '', xy=(0.5, y - row_h * 0.55),
                        xytext=(0.5, y - row_h * 0.45),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8)
                    )

            if truncated:
                y_more = 1.0 - (MAX_VISIBLE + 1) * row_h
                ax_queue.text(
                    0.5, y_more,
                    f'… {len(display) - MAX_VISIBLE} more',
                    ha='center', va='center', fontsize=8, color='gray',
                    transform=ax_queue.transAxes
                )

        # ------------------------------------------------------------------
        # Core update — grid + tree + queue, driven each frame
        # ------------------------------------------------------------------
        def update(frame):
            done     = frame >= len(events)
            next_pop = None   # node highlighted at top of queue this frame

            # Compute current iteration boundary once — reused by grid and tree.
            # iter_start is the index of the first event after the most recent
            # new_iteration marker up to this frame.
            visible_all = events[:min(frame + 1, len(events))]
            iter_start  = 0
            for i, (t, _) in enumerate(visible_all):
                if t == 'new_iteration':
                    iter_start = i + 1
            current_iter_events = visible_all[iter_start:]

            # ---- Open-list mirror update ----
            if not done:
                event_type, payload = events[frame]

                if event_type == 'new_iteration':
                    open_list.clear()
                    iter_parent[0] = None
                elif event_type == 'parent_snapshot':
                    iter_parent[0] = payload
                elif event_type in ('discover', 'push'):
                    node_id = payload[0] if isinstance(payload, tuple) else payload
                    _ol_add(node_id)
                elif event_type == 'visit':
                    next_pop = payload
                    _ol_remove(payload)

            # ---- Grid panel ----
            if not done:
                event_type, payload = events[frame]

                if event_type == 'new_iteration':
                    # Reset grid and node_state regardless of agent type
                    color_grid[:] = base_grid.copy()
                    node_state.clear()
                    grid_title.set_text(
                        f"Grid View — Iteration depth limit {payload}")
                    grid_title.set_color('steelblue')

                elif event_type == 'parent_snapshot':
                    pass  # handled in open-list mirror; no grid action needed

                elif self._shows_discovered:
                    node_id = payload
                    r, c = (self.grid.G.nodes[node_id]['row'],
                            self.grid.G.nodes[node_id]['col'])
                    if event_type == 'discover':
                        node_state[node_id] = 'discovered'
                        color_grid[r][c] = 5
                    elif event_type == 'visit':
                        for n, s in list(node_state.items()):
                            if s == 'agent':
                                node_state[n] = 'visited'
                                nr, nc = (self.grid.G.nodes[n]['row'],
                                          self.grid.G.nodes[n]['col'])
                                color_grid[nr][nc] = 2
                        node_state[node_id] = 'agent'
                        color_grid[r][c] = 3
                        grid_title.set_text(
                            f"Grid View — step {frame + 1}/{len(events)}")
                        grid_title.set_color('gray')

                else:
                    if event_type == 'visit':
                        node_id = payload
                        prev_visit = next(
                            (n for t, n in reversed(current_iter_events[:-1])
                             if t == 'visit'), None)
                        if prev_visit is not None:
                            node_state[prev_visit] = 'visited'
                            pr, pc = (self.grid.G.nodes[prev_visit]['row'],
                                      self.grid.G.nodes[prev_visit]['col'])
                            color_grid[pr][pc] = 2
                        node_state[node_id] = 'agent'
                        r, c = (self.grid.G.nodes[node_id]['row'],
                                self.grid.G.nodes[node_id]['col'])
                        color_grid[r][c] = 3
                        grid_title.set_text(
                            f"Grid View — step {frame + 1}/{len(events)}")
                        grid_title.set_color('gray')

            else:
                if self.agent.found:
                    for n in path:
                        nr, nc = (self.grid.G.nodes[n]['row'],
                                  self.grid.G.nodes[n]['col'])
                        color_grid[nr][nc] = 4
                    grid_title.set_text("Grid View — Path Found ✓")
                    grid_title.set_color('green')
                else:
                    grid_title.set_text("Grid View — No Path Found ✗")
                    grid_title.set_color('red')
            im.set_data(color_grid)

            # ---- Tree panel ----
            ax_tree.cla()
            ax_tree.axis('off')

            # current_iter_events already computed at top of update()
            visited_set  = {n for t, n in current_iter_events if t == 'visit'}
            discover_set = ({n for t, n in current_iter_events if t == 'discover'}
                            if self._shows_discovered else set())

            # Build the parent map for tree construction this frame.
            # DFS/IDDFS: reconstruct live from push tuple events since
            #   agent.parent is overwritten each iteration.
            # All other agents: agent.parent is stable and correct throughout —
            #   discover events don't carry parent info so we read it directly.
            from agent import IDDFSAgent as _IDDFS, DFSAgent as _DFS
            if isinstance(self.agent, (_DFS, _IDDFS)):
                live_parent = {self.agent.grid.start: None}
                for t, payload in current_iter_events:
                    if t == 'push' and isinstance(payload, tuple):
                        neighbor, par = payload
                        if par is not None:
                            live_parent[neighbor] = par
            else:
                live_parent = self.agent.parent

            # Keep hover_parent in sync with what's visible on the grid
            hover_parent.clear()
            hover_parent.update(live_parent)

            # Build tree from current iteration events using live_parent.
            # DFS/IDDFS: only 'visit' events — no discover state.
            # BFS/Greedy/A*: include 'discover' events so frontier nodes appear.
            tree_curr = nx.DiGraph()
            tree_curr.add_node(self.agent.grid.start)
            relevant = ('visit',) if not self._shows_discovered else ('visit', 'discover')
            for t, n in current_iter_events:
                if t in relevant and isinstance(n, int):
                    par = live_parent.get(n)
                    if par is not None:
                        tree_curr.add_edge(par, n)

            curr = (next((n for t, n in reversed(current_iter_events)
                          if t == 'visit'), None) if not done else None)

            # Compute layout for tree_curr. For IDDFS earlier iterations may
            # contain nodes that never made it into the final agent.parent, so
            # they have no entry in the pre-computed global pos. Recompute
            # layout from tree_curr itself so every node has a position.
            if tree_curr.number_of_nodes() > 1:
                iter_pos = _hierarchy_pos(tree_curr, self.agent.grid.start)
            else:
                iter_pos = {self.agent.grid.start: (0, 0)}
            vis_pos = iter_pos

            positioned = [n for n in tree_curr.nodes() if n in vis_pos]
            positioned_set = set(positioned)

            node_colors = self._node_colors_for_tree(
                tree_curr, visited_set, discover_set, curr, path_set, done)
            # node_colors is ordered by tree_curr.nodes() — filter to positioned
            node_colors = [c for n, c in zip(tree_curr.nodes(), node_colors)
                           if n in positioned_set]
            edge_colors = [
                'mediumseagreen'
                if (done and self.agent.found and u in path_set and v in path_set)
                else 'gray'
                for u, v in tree_curr.edges()
                if u in positioned_set and v in positioned_set
            ]
            positioned_edges = [(u, v) for u, v in tree_curr.edges()
                                if u in positioned_set and v in positioned_set]

            nx.draw_networkx_nodes(tree_curr, vis_pos, ax=ax_tree,
                                   nodelist=positioned,
                                   node_color=node_colors, node_size=500,
                                   edgecolors='black', linewidths=0.8)
            nx.draw_networkx_edges(tree_curr, vis_pos, ax=ax_tree,
                                   edgelist=positioned_edges,
                                   edge_color=edge_colors, arrows=True,
                                   arrowsize=12, width=1.5)
            nx.draw_networkx_labels(tree_curr, vis_pos, ax=ax_tree,
                                    labels={n: self._node_label(n)
                                            for n in positioned},
                                    font_size=7, font_weight='bold')

            if done:
                ax_tree.set_title(
                    "Tree View — Path Found ✓" if self.agent.found
                    else "Tree View — No Path Found ✗",
                    fontsize=13, fontweight='bold',
                    color='green' if self.agent.found else 'red')
            else:
                ax_tree.set_title(
                    f"Tree View — step {frame + 1}/{len(events)}",
                    fontsize=13, fontweight='bold', color='gray')
            ax_tree.legend(handles=_tree_legend(self._shows_discovered),
                           loc='upper right', fontsize=8)

            # ---- Queue panel ----
            _draw_queue(next_pop)

        # ------------------------------------------------------------------
        # FuncAnimation wrapper
        # ------------------------------------------------------------------
        def _anim_update(_frame):
            if state['playing'] and state['frame'] < total_frames:
                update(state['frame'])
                state['frame'] += 1
            fig.canvas.draw_idle()

        # ------------------------------------------------------------------
        # Control callbacks
        # ------------------------------------------------------------------
        def on_play_pause(_event):
            if state['playing']:
                _anim.pause()
                state['playing'] = False
                btn_playpause.label.set_text('Play')
            else:
                _anim.resume()
                state['playing'] = True
                btn_playpause.label.set_text('Pause')
            fig.canvas.draw_idle()

        def on_step(_event):
            if state['playing']:
                _anim.pause()
                state['playing'] = False
                btn_playpause.label.set_text('Play')
            if state['frame'] < total_frames:
                update(state['frame'])
                state['frame'] += 1
                fig.canvas.draw_idle()

        def on_restart(_event):
            color_grid[:] = base_grid.copy()
            node_state.clear()
            open_list.clear()
            hover_parent.clear()
            iter_parent[0] = None
            state['frame'] = 0
            state['playing'] = True
            btn_playpause.label.set_text('Pause')
            update(0)
            state['frame'] = 1
            _anim.resume()
            fig.canvas.draw_idle()

        def on_speed(_val):
            interval = int(sld_speed.val)
            _anim._interval = interval
            _anim.event_source.interval = interval

        btn_playpause.on_clicked(on_play_pause)
        btn_step.on_clicked(on_step)
        btn_restart.on_clicked(on_restart)
        sld_speed.on_changed(on_speed)

        # ------------------------------------------------------------------
        # Hover callback
        # ------------------------------------------------------------------
        def on_hover(event):
            if event.inaxes is not ax_grid:
                info_text.set_text('')
                fig.canvas.draw_idle()
                return

            col = int(round(event.xdata)) if event.xdata is not None else -1
            row = int(round(event.ydata)) if event.ydata is not None else -1

            if not (0 <= row < self.grid.size and 0 <= col < self.grid.size):
                info_text.set_text('')
                fig.canvas.draw_idle()
                return

            node_id = row * self.grid.size + col

            # Gate on node_state — reflects exactly what's currently colored
            # on the grid regardless of which iteration we're in
            if node_id not in node_state:
                info_text.set_text('')
                fig.canvas.draw_idle()
                return

            lines = [f'Node ({row},{col})']

            # Use hover_parent (current iteration) not agent.parent (final iteration)
            par = hover_parent.get(node_id)
            if node_id == self.agent.grid.start or par is None:
                lines.append('parent: start')
            else:
                pr, pc = par // self.grid.size, par % self.grid.size
                lines.append(f'parent: ({pr},{pc})')

            order = self.agent.visit_order.get(node_id)
            lines.append(f'visit order: {order}' if order is not None
                         else 'visit order: pending (frontier)')

            if h_cost is not None and node_id in h_cost:
                lines.append(f'h(n): {h_cost[node_id]}')

            if g_cost is not None and node_id in g_cost:
                g = g_cost[node_id]
                h = h_cost.get(node_id, 0)
                lines.append(f'g(n): {g}')
                lines.append(f'f(n): {g + h}')

            info_text.set_text('    '.join(lines))
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_hover)

        # ------------------------------------------------------------------
        # Launch
        # ------------------------------------------------------------------
        _anim = FuncAnimation(fig, _anim_update, interval=250,
                              cache_frame_data=False, repeat=False)
        plt.show()

    # ------------------------------------------------------------------
    # Graph visualization (CityGraph / RandomGraph)
    # ------------------------------------------------------------------

    def _show_all_graph(self, metrics=None):
        """
        Animate a CityGraph or RandomGraph search on the actual network.

        Layout
        ------
        Left (wide)  : the full graph — edges gray, nodes color-coded by
                       state, visited edges drawn in blue as traversed,
                       path edges highlighted green on completion.
        Right        : live open list (queue / stack / priority queue).
        Bottom strip : Play/Pause, Step, Restart, Speed slider.
        Info box     : hover details — city name, parent, visit order,
                       g/h/f costs where available.

        No tree panel — the graph IS the search space, so watching nodes
        and edges light up on the real map is more informative than an
        abstract hierarchy.
        """
        global _anim
        from agent import BFSAgent, DFSAgent, IDDFSAgent, GreedyAgent, AStarAgent
        from matplotlib.widgets import Button, Slider

        env      = self.grid
        G        = env.G
        pos      = {n: (float(xy[0]), float(xy[1])) for n, xy in env.pos.items()}
        labels   = env.labels

        path      = self.agent.reconstruct_path()
        path_set  = set(path)
        path_cost = (sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                     if len(path) > 1 else 0.0)
        is_city   = hasattr(G.nodes[env.start], '__contains__') and 'lat' in G.nodes[env.start]

        events       = self.agent.events
        hold_frames  = 20
        total_frames = len(events) + hold_frames
        state        = {'frame': 0, 'playing': True, 'metrics_drawn': False}

        self.agent.visit_sequence()
        g_cost = getattr(self.agent, 'g_cost', None)
        h_cost = getattr(self.agent, 'h_cost', None)

        # Per-node display state
        node_state   = {n: 'unvisited' for n in G.nodes()}
        hover_parent = {}

        # Track which edges have been traversed (for progressive edge coloring)
        traversed_edges = set()   # set of frozenset({u,v})
        path_edges_drawn = []     # matplotlib artists added at completion

        # ---- Open-list mirror ------------------------------------------------
        open_list = []
        if isinstance(self.agent, BFSAgent):
            _ol_title = 'Queue  (front → back)'
            def _ol_add(n):
                open_list.append(n)
            def _ol_remove(n):
                if n in open_list: open_list.remove(n)
            def _ol_label(n):
                return labels.get(n, str(n))

        elif isinstance(self.agent, (DFSAgent, IDDFSAgent)):
            _ol_title = 'Stack  (top → bottom)'
            def _ol_add(n):
                open_list.append(n)
            def _ol_remove(n):
                if n in open_list: open_list.remove(n)
            def _ol_label(n):
                return labels.get(n, str(n))

        elif isinstance(self.agent, GreedyAgent):
            _ol_title = 'Open List  ↑ h(n)'
            def _ol_add(n):
                if n in open_list: open_list.remove(n)
                open_list.append(n)
                open_list.sort(key=lambda x: h_cost.get(x, float('inf')))
            def _ol_remove(n):
                if n in open_list: open_list.remove(n)
            def _ol_label(n):
                h   = h_cost.get(n)
                lbl = labels.get(n, str(n))
                return f'{lbl}  h={h:.1f}' if isinstance(h, float) else lbl

        else:  # AStarAgent
            _ol_title = 'Open List  ↑ f(n)'
            def _f(n):
                return (g_cost.get(n) or 0) + (h_cost.get(n) or 0)
            def _ol_add(n):
                if n in open_list: open_list.remove(n)
                open_list.append(n)
                open_list.sort(key=_f)
            def _ol_remove(n):
                if n in open_list: open_list.remove(n)
            def _ol_label(n):
                g   = g_cost.get(n)
                h   = h_cost.get(n)
                lbl = labels.get(n, str(n))
                if isinstance(g, float) and isinstance(h, float):
                    return f'{lbl}  {g:.1f}+{h:.1f}={g+h:.1f}'
                return lbl

        MAX_VISIBLE = 16

        # ---- Figure: 3-panel (graph wide | open list | metrics) ---------------
        fig = plt.figure(figsize=(24, 9))
        gs  = fig.add_gridspec(1, 3, width_ratios=[5, 1, 1.4],
                               left=0.03, right=0.97,
                               bottom=0.18, top=0.93, wspace=0.18)
        ax_graph   = fig.add_subplot(gs[0])
        ax_queue   = fig.add_subplot(gs[1])
        ax_metrics = fig.add_subplot(gs[2])
        ax_metrics.axis('off')

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()

        agent_name = type(self.agent).__name__.replace('Agent', '')
        fig.suptitle(f"Search Visualization — {agent_name}",
                     fontsize=18, fontweight='bold')

        # ---- Info text box (bottom-left) -------------------------------------
        ax_info = fig.add_axes([0.03, 0.07, 0.55, 0.08])
        ax_info.axis('off')
        info_text = ax_info.text(
            0.01, 0.95, '',
            transform=ax_info.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9)
        )

        # ---- Controls --------------------------------------------------------
        ax_restart   = fig.add_axes([0.05, 0.01, 0.07, 0.05])
        ax_step      = fig.add_axes([0.13, 0.01, 0.07, 0.05])
        ax_playpause = fig.add_axes([0.21, 0.01, 0.07, 0.05])
        ax_speed     = fig.add_axes([0.42, 0.015, 0.35, 0.03])

        btn_restart   = Button(ax_restart,   'Restart', color='0.85', hovercolor='0.75')
        btn_step      = Button(ax_step,      'Step',    color='0.85', hovercolor='0.75')
        btn_playpause = Button(ax_playpause, 'Pause',   color='0.85', hovercolor='0.75')
        sld_speed     = Slider(ax_speed, 'ms/frame', 50, 1000,
                               valinit=250, valstep=50, color='steelblue')

        # ---- Static graph drawing --------------------------------------------
        node_list = list(G.nodes())
        node_xy   = np.array([pos[n] for n in node_list])

        # All edges — drawn once in light gray
        nx.draw_networkx_edges(G, pos, ax=ax_graph,
                               width=1.2, edge_color='#dddddd', alpha=0.9)

        # Edge weight labels
        ew_labels = {(u, v): f"{d['weight']:.0f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=ew_labels,
                                     ax=ax_graph, font_size=6,
                                     font_color='#aaaaaa', label_pos=0.38)

        # Node circles — face color updated each frame via PathCollection
        node_scatter = nx.draw_networkx_nodes(
            G, pos, ax=ax_graph, nodelist=node_list,
            node_color=['#f0f0f0'] * len(node_list),
            node_size=700, edgecolors='#555555', linewidths=1.0
        )

        # Labels offset above nodes
        lbl_pos = {n: (x, y + 0.42) for n, (x, y) in pos.items()}
        nx.draw_networkx_labels(G, lbl_pos, labels=labels,
                                ax=ax_graph, font_size=7.5, font_weight='bold')

        # Start / goal permanent markers on top
        sx, sy = pos[env.start]
        gx, gy = pos[env.goal]
        ax_graph.scatter([sx], [sy], c='gold',       s=800, zorder=8,
                         edgecolors='black', linewidths=1.5, marker='^')
        ax_graph.scatter([gx], [gy], c='dodgerblue', s=800, zorder=8,
                         edgecolors='black', linewidths=1.5, marker='X')

        graph_title = ax_graph.set_title(
            f"Searching…  (start: {labels[env.start]}  →  goal: {labels[env.goal]})",
            fontsize=13, fontweight='bold', color='gray')
        ax_graph.axis('off')

        # Legend
        _COLOR = {
            'unvisited':  '#f0f0f0',
            'discovered': 'lightyellow',
            'agent':      'tomato',
            'visited':    'lightblue',
        }
        leg = [
            mpatches.Patch(facecolor='#f0f0f0',       edgecolor='#555', label='Unvisited'),
            mpatches.Patch(facecolor='lightblue',      edgecolor='#555', label='Visited'),
            mpatches.Patch(facecolor='tomato',         edgecolor='#555', label='Current'),
            mpatches.Patch(facecolor='mediumseagreen', edgecolor='#555', label='Path'),
            plt.scatter([], [], marker='^', c='gold',       s=80, edgecolors='black', label='Start'),
            plt.scatter([], [], marker='X', c='dodgerblue', s=80, edgecolors='black', label='Goal'),
        ]
        if self._shows_discovered:
            leg.insert(2, mpatches.Patch(facecolor='lightyellow',
                                          edgecolor='gray', label='Discovered'))
        ax_graph.legend(handles=leg, loc='upper left', fontsize=8,
                        framealpha=0.9, edgecolor='#ccc')

        # Container for traversal-edge artists (drawn progressively)
        traversal_artists = {}   # frozenset({u,v}) → matplotlib artist

        # ---- Helpers ---------------------------------------------------------
        def _node_colors(done=False):
            colors = []
            for n in node_list:
                if done and n in path_set:
                    colors.append('mediumseagreen')
                else:
                    colors.append(_COLOR.get(node_state[n], '#f0f0f0'))
            return colors

        def _draw_traversal_edge(u, v):
            """Draw a single traversed edge in steelblue."""
            key = frozenset((u, v))
            if key not in traversal_artists and G.has_edge(u, v):
                arts = nx.draw_networkx_edges(
                    G, pos, ax=ax_graph, edgelist=[(u, v)],
                    width=2.2, edge_color='steelblue', alpha=0.55,
                    arrows=False
                )
                if arts is not None:
                    traversal_artists[key] = arts

        # ---- Metrics summary panel -------------------------------------------
        def _draw_metrics_panel(m):
            """
            Draw the metrics summary in the dedicated ax_metrics panel.
            Called once when the animation completes.  Because it uses its
            own axes (not the graph canvas), it never overlaps the graph and
            is unaffected by animation restarts.

            m : MetricsResult
            """
            ax_metrics.cla()
            ax_metrics.axis('off')

            opt_sym = '✓' if m.is_optimal else ('✗' if m.is_optimal is False else '?')
            opt_col = 'green' if m.is_optimal else ('tomato' if m.is_optimal is False else 'gray')
            unit    = 'km' if 'lat' in env.G.nodes[env.start] else 'units'
            agent_short = m.agent_name.replace('Agent', '')

            ax_metrics.set_title(f'{agent_short} — Results',
                                 fontsize=10, fontweight='bold', color='steelblue', pad=6)

            # Build rows as (label, value, value_color) tuples
            rows = [
                ('Runtime',      f'{m.runtime_s*1000:.2f} ms',   'black'),
                ('Process mem',  f'{m.process_peak_kb:.1f} KB',  'black'),
                ('Frontier peak',f'{m.algorithmic_peak_frontier}','black'),
                ('Explored',     f'{m.algorithmic_explored}',     'black'),
                (None, None, None),   # spacer
                ('Generated',    f'{m.nodes_generated}',          'black'),
                ('Expanded',     f'{m.nodes_expanded}',           'black'),
                ('Branch avg',   f'{m.branching_factor_avg:.2f}', 'black'),
                ('Branch max',   f'{m.branching_factor_max}',     'black'),
                ('Depth',        f'{m.solution_depth}',           'black'),
                ('Path cost',    f'{m.path_cost:.1f} {unit}',     'black'),
                (None, None, None),   # spacer
                ('Optimal',      f'{opt_sym}',                    opt_col),
            ]

            n_rows   = len(rows)
            row_h    = 1.0 / (n_rows + 1)
            lbl_x    = 0.05
            val_x    = 0.97

            # Divider line under title
            ax_metrics.plot([0.0, 1.0], [1.0 - row_h * 0.4]*2, color='#cccccc',
                            linewidth=0.8, transform=ax_metrics.transAxes, clip_on=False)

            for i, (lbl, val, col) in enumerate(rows):
                y = 1.0 - (i + 1) * row_h
                if lbl is None:
                    # Draw a thin separator line
                    ax_metrics.plot([0.0, 1.0], [y + row_h * 0.3]*2, color='#eeeeee',
                                   linewidth=0.6, transform=ax_metrics.transAxes, clip_on=False)
                    continue
                ax_metrics.text(lbl_x, y, lbl,
                                transform=ax_metrics.transAxes,
                                fontsize=8, va='center', ha='left',
                                color='#555555', family='monospace')
                ax_metrics.text(val_x, y, val,
                                transform=ax_metrics.transAxes,
                                fontsize=8, va='center', ha='right',
                                color=col, family='monospace', fontweight='bold')

            # Optimality reason as small text at bottom
            ax_metrics.text(0.5, 0.01, m.optimality_reason,
                            transform=ax_metrics.transAxes,
                            fontsize=6.5, va='bottom', ha='center',
                            color=opt_col, style='italic', wrap=True)

            fig.canvas.draw_idle()

        # ---- Queue panel helper ----------------------------------------------
        def _draw_queue(next_pop):
            ax_queue.cla()
            ax_queue.axis('off')
            ax_queue.set_title(_ol_title, fontsize=9, fontweight='bold',
                               color='steelblue', pad=4)
            if not open_list:
                ax_queue.text(0.5, 0.5, '(empty)', ha='center', va='center',
                              fontsize=9, color='gray',
                              transform=ax_queue.transAxes)
                return

            is_stack = isinstance(self.agent, (DFSAgent, IDDFSAgent))
            display  = list(reversed(open_list)) if is_stack else list(open_list)
            truncated = len(display) > MAX_VISIBLE
            visible   = display[:MAX_VISIBLE]
            n_rows    = len(visible) + (1 if truncated else 0)
            row_h     = 1.0 / (n_rows + 1)

            for i, nid in enumerate(visible):
                y       = 1.0 - (i + 1) * row_h
                is_next = (nid == next_pop)
                bg   = 'tomato'  if is_next else 'lightyellow'
                ec   = 'darkred' if is_next else '#aaa'
                fw   = 'bold'    if is_next else 'normal'
                fc   = 'white'   if is_next else 'black'
                ax_queue.text(0.5, y, _ol_label(nid),
                              ha='center', va='center', fontsize=7.5,
                              fontweight=fw, color=fc,
                              transform=ax_queue.transAxes,
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor=bg, edgecolor=ec, linewidth=0.8))
                if i < len(visible) - 1:
                    ax_queue.annotate(
                        '', xy=(0.5, y - row_h * 0.55),
                        xytext=(0.5, y - row_h * 0.45),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8))

            if truncated:
                y_more = 1.0 - (MAX_VISIBLE + 1) * row_h
                ax_queue.text(0.5, y_more,
                              f'… {len(display) - MAX_VISIBLE} more',
                              ha='center', va='center', fontsize=8, color='gray',
                              transform=ax_queue.transAxes)

        # ---- Core update -----------------------------------------------------
        def update(frame):
            nonlocal path_edges_drawn
            done     = frame >= len(events)
            next_pop = None
            prev_agent_node = None

            if not done:
                event_type, payload = events[frame]

                # -- Open-list mirror --
                if event_type == 'new_iteration':
                    open_list.clear()
                elif event_type == 'parent_snapshot':
                    pass
                elif event_type in ('discover', 'push'):
                    nid = payload[0] if isinstance(payload, tuple) else payload
                    _ol_add(nid)
                elif event_type == 'visit':
                    next_pop = payload
                    _ol_remove(payload)

                # -- Node / edge state --
                if event_type == 'new_iteration':
                    for n in node_state: node_state[n] = 'unvisited'
                    traversal_artists.clear()
                    hover_parent.clear()
                    graph_title.set_text(f"Iteration — depth limit {payload}")
                    graph_title.set_color('steelblue')

                elif event_type == 'parent_snapshot':
                    pass

                elif self._shows_discovered:
                    nid = payload
                    if event_type == 'discover':
                        if node_state[nid] == 'unvisited':
                            node_state[nid] = 'discovered'
                        par = self.agent.parent.get(nid)
                        if par is not None:
                            _draw_traversal_edge(par, nid)
                    elif event_type == 'visit':
                        for n, s in list(node_state.items()):
                            if s == 'agent':
                                prev_agent_node = n
                                node_state[n] = 'visited'
                        node_state[nid] = 'agent'
                        graph_title.set_text(
                            f"{labels[env.start]} → {labels[env.goal]}"
                            f"   step {frame+1}/{len(events)}")
                        graph_title.set_color('gray')

                else:  # DFS-like — only visit events
                    if event_type == 'visit':
                        nid = payload
                        for n, s in list(node_state.items()):
                            if s == 'agent':
                                prev_agent_node = n
                                node_state[n] = 'visited'
                        node_state[nid] = 'agent'
                        # Draw edge from parent to current
                        par = self.agent.parent.get(nid)
                        if par is not None:
                            _draw_traversal_edge(par, nid)
                        graph_title.set_text(
                            f"{labels[env.start]} → {labels[env.goal]}"
                            f"   step {frame+1}/{len(events)}")
                        graph_title.set_color('gray')

                # Update hover parent map
                if isinstance(self.agent, (DFSAgent, IDDFSAgent)):
                    live_parent = {env.start: None}
                    for t, pl in events[:frame + 1]:
                        if t == 'push' and isinstance(pl, tuple):
                            nb, par = pl
                            if par is not None:
                                live_parent[nb] = par
                    hover_parent.clear()
                    hover_parent.update(live_parent)
                else:
                    hover_parent.clear()
                    hover_parent.update(self.agent.parent)

                node_scatter.set_facecolor(_node_colors(done=False))

            else:
                # Search complete — draw final state
                for art in path_edges_drawn:
                    try: art.remove()
                    except Exception: pass
                path_edges_drawn = []

                node_scatter.set_facecolor(_node_colors(done=True))

                if self.agent.found and len(path) > 1:
                    pes = nx.draw_networkx_edges(
                        G, pos, ax=ax_graph,
                        edgelist=list(zip(path[:-1], path[1:])),
                        width=5.0, edge_color='mediumseagreen',
                        arrows=True, arrowsize=22, arrowstyle='->'
                    )
                    if pes is not None:
                        try: path_edges_drawn = list(pes)
                        except TypeError: path_edges_drawn = [pes]

                    unit = 'km' if 'lat' in G.nodes[env.start] else 'units'
                    graph_title.set_text(
                        f"Path found ✓   {labels[env.start]} → {labels[env.goal]}"
                        f"   {len(path)-1} hops   cost: {path_cost:.1f} {unit}")
                    graph_title.set_color('green')
                else:
                    graph_title.set_text(
                        f"No path found ✗   {labels[env.start]} → {labels[env.goal]}")
                    graph_title.set_color('red')

                # Draw metrics summary panel if metrics were provided
                if metrics is not None and not state.get('metrics_drawn'):
                    state['metrics_drawn'] = True
                    _draw_metrics_panel(metrics)

            _draw_queue(next_pop)
            fig.canvas.draw_idle()

        # ---- Animation controls ----------------------------------------------
        def _anim_update(_frame):
            if state['playing'] and state['frame'] < total_frames:
                update(state['frame'])
                state['frame'] += 1

        def on_play_pause(_event):
            if state['playing']:
                _anim.pause(); state['playing'] = False
                btn_playpause.label.set_text('Play')
            else:
                _anim.resume(); state['playing'] = True
                btn_playpause.label.set_text('Pause')
            fig.canvas.draw_idle()

        def on_step(_event):
            if state['playing']:
                _anim.pause(); state['playing'] = False
                btn_playpause.label.set_text('Play')
            if state['frame'] < total_frames:
                update(state['frame']); state['frame'] += 1
                fig.canvas.draw_idle()

        def on_restart(_event):
            for n in node_state: node_state[n] = 'unvisited'
            traversal_artists.clear()
            for art in path_edges_drawn:
                try: art.remove()
                except Exception: pass
            open_list.clear()
            hover_parent.clear()
            state['frame'] = 0; state['playing'] = True; state['metrics_drawn'] = False
            btn_playpause.label.set_text('Pause')
            update(0); state['frame'] = 1
            _anim.resume(); fig.canvas.draw_idle()

        def on_speed(_val):
            _anim._interval = int(sld_speed.val)
            _anim.event_source.interval = int(sld_speed.val)

        btn_playpause.on_clicked(on_play_pause)
        btn_step.on_clicked(on_step)
        btn_restart.on_clicked(on_restart)
        sld_speed.on_changed(on_speed)

        # ---- Hover -----------------------------------------------------------
        def on_hover(event):
            if event.inaxes is not ax_graph or event.xdata is None:
                info_text.set_text('')
                fig.canvas.draw_idle()
                return

            click = np.array([event.xdata, event.ydata])
            dists = np.linalg.norm(node_xy - click, axis=1)
            nid   = node_list[int(np.argmin(dists))]

            if node_state[nid] == 'unvisited':
                info_text.set_text('')
                fig.canvas.draw_idle()
                return

            lbl   = labels.get(nid, str(nid))
            lines = [f'City: {lbl}']

            par = hover_parent.get(nid)
            if nid == env.start or par is None:
                lines.append('via: start')
            else:
                lines.append(f'via: {labels.get(par, str(par))}')

            order = self.agent.visit_order.get(nid)
            lines.append(f'visit #{order}' if order else 'frontier')

            if h_cost and nid in h_cost:
                lines.append(f'h={h_cost[nid]:.1f}')
            if g_cost and nid in g_cost:
                g = g_cost[nid]
                h = (h_cost or {}).get(nid, 0.0)
                lines.append(f'g={g:.1f}  f={g+h:.1f}')

            info_text.set_text('    '.join(lines))
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_hover)

        # ---- Launch ----------------------------------------------------------
        _anim = FuncAnimation(fig, _anim_update, interval=250,
                              cache_frame_data=False, repeat=False)
        plt.show()