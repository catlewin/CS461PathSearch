"""
environment_generation.py
-------------------------
Defines the Grid class, which builds and owns a NetworkX DiGraph representation
of a square grid environment for pathfinding agents.

All setup — node creation, obstacle placement, edge wiring, and start/goal
selection — happens in __init__, so callers get a fully ready environment
from a single instantiation.

Classes
-------
Grid
    Square grid environment backed directly by a NetworkX DiGraph.

Functions
---------
get_node_labels(num_vertices)
    Generates alphabetical labels (A, B, ..., Z, AA, AB, ...) for graph
    visualization. Kept as a standalone utility since it depends only on
    a vertex count, not on any Grid state.
"""

import random
import string
import matplotlib.pyplot as plt
import networkx as nx


# Cardinal directions in clockwise order starting from right.
# Stored as (dr, dc, name) so edges are tagged with their direction,
# giving agents an explicit stable traversal order independent of
# NetworkX internal edge ordering.
_DIRECTIONS_CW = [
    ( 0,  1, 'right'),
    ( 1,  0, 'down'),
    ( 0, -1, 'left'),
    (-1,  0, 'up'),
]

# Maps direction name → clockwise priority index (right=0, down=1, left=2, up=3).
_CW_ORDER = {name: i for i, (_, _, name) in enumerate(_DIRECTIONS_CW)}


def get_node_labels(num_vertices):
    """
    Generate alphabetical labels for graph nodes (A–Z, then AA, AB, ...).

    Parameters
    ----------
    num_vertices : int
        Total number of nodes to label.

    Returns
    -------
    dict
        Maps integer node ID → alphabetical label string.
    """
    labels = {}
    alphabet = string.ascii_uppercase
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


class Grid:
    """
    Square grid environment backed directly by a NetworkX DiGraph.

    Nodes represent grid cells (flattened to integer IDs: r * size + c).
    Each node carries 'row', 'col', and 'passable' attributes. Directed
    edges connect passable nodes to their cardinal passable neighbors.

    Parameters
    ----------
    size : int, optional
        Side length of the square grid (default 10), giving size² nodes total.

    Attributes
    ----------
    size : int
        Side length of the grid.
    obstacle_pct : float
        Fraction of cells randomly designated as obstacles (sampled from [0.2, 0.3]).
    G : nx.DiGraph
        The graph representation of the grid.
    start : int
        Flat node ID of the start cell (guaranteed passable).
    goal : int
        Flat node ID of the goal cell (guaranteed passable).
    """

    def __init__(self, size=10):
        self.size = size
        self.obstacle_pct = random.uniform(0.20, 0.30)
        self.G = nx.DiGraph()

        self._build_nodes()
        self._place_obstacles()
        self._build_edges()
        self.start, self.goal = self._pick_start_and_goal()

    # ------------------------------------------------------------------
    # Internal setup helpers
    # ------------------------------------------------------------------

    def _build_nodes(self):
        """Add all size² nodes to the graph, each initially passable."""
        for r in range(self.size):
            for c in range(self.size):
                node_id = r * self.size + c
                self.G.add_node(node_id, row=r, col=c, passable=True)

    def _place_obstacles(self):
        """Mark a random obstacle_pct fraction of nodes as non-passable."""
        num_obstacles = int(self.size ** 2 * self.obstacle_pct)
        obstacle_ids = random.sample(list(self.G.nodes), num_obstacles)
        for node_id in obstacle_ids:
            self.G.nodes[node_id]['passable'] = False

    def _build_edges(self):
        """
        Wire directed edges between passable cardinal neighbors.

        Each edge is tagged with a 'direction' attribute (one of 'right',
        'down', 'left', 'up') so agents can retrieve neighbors in a
        guaranteed clockwise order via neighbors_clockwise(), independent
        of NetworkX internal edge ordering.
        """
        for r in range(self.size):
            for c in range(self.size):
                node_id = r * self.size + c
                if not self.G.nodes[node_id]['passable']:
                    continue
                for dr, dc, direction in _DIRECTIONS_CW:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        neighbor_id = nr * self.size + nc
                        if self.G.nodes[neighbor_id]['passable']:
                            self.G.add_edge(node_id, neighbor_id, direction=direction)

    def _pick_start_and_goal(self):
        """
        Randomly select two passable nodes as start and goal.

        Returns
        -------
        tuple[int, int]
            (start_id, goal_id) flat node IDs.

        Raises
        ------
        ValueError
            If fewer than two passable nodes exist.
        """
        passable = [n for n, d in self.G.nodes(data=True) if d['passable']]
        if len(passable) < 2:
            raise ValueError("Grid has fewer than 2 open cells; cannot place start and goal.")
        start, goal = random.sample(passable, 2)
        return start, goal

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def node_id(self, row, col):
        """Convert (row, col) coordinates to a flat node ID."""
        return row * self.size + col

    def neighbors_clockwise(self, node_id):
        """
        Return passable neighbors of node_id sorted clockwise from right.

        Sorting is done on the 'direction' edge attribute rather than
        relying on NetworkX iteration order, guaranteeing right → down
        → left → up regardless of insertion order.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        list[int]
            Neighbor node IDs in clockwise order.
        """
        return sorted(
            self.G.neighbors(node_id),
            key=lambda n: _CW_ORDER[self.G[node_id][n]['direction']]
        )

    def visualize(self):
        """
        Render the grid graph with color-coded nodes and a legend.

        Node colors:
            - Light green : passable (open)
            - Tomato      : obstacle
            - Gold        : start
            - Dodger blue : goal
        """
        plt.figure(figsize=(8, 8))

        pos = {n: (d['col'], -d['row']) for n, d in self.G.nodes(data=True)}

        passable = [n for n, d in self.G.nodes(data=True)
                    if d['passable'] and n not in (self.start, self.goal)]
        blocked  = [n for n, d in self.G.nodes(data=True) if not d['passable']]

        nx.draw_networkx_nodes(self.G, pos, nodelist=passable,   node_color='lightgreen', node_size=500, edgecolors='black')
        nx.draw_networkx_nodes(self.G, pos, nodelist=blocked,    node_color='tomato',     node_size=500, edgecolors='black')
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.start], node_color='gold',     node_size=600, edgecolors='black')
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.goal],  node_color='dodgerblue', node_size=600, edgecolors='black')
        nx.draw_networkx_edges(self.G, pos, width=1.5, edge_color='gray')

        labels = get_node_labels(self.size ** 2)
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=9, font_weight='bold')

        plt.scatter([], [], c='lightgreen',  label='Open',    edgecolors='black', s=100)
        plt.scatter([], [], c='tomato',      label='Blocked', edgecolors='black', s=100)
        plt.scatter([], [], c='gold',        label='Start',   edgecolors='black', s=100)
        plt.scatter([], [], c='dodgerblue',  label='Goal',    edgecolors='black', s=100)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15), borderaxespad=0)

        plt.title(f"Grid Graph ({self.size}×{self.size})", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()