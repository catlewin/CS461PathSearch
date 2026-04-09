"""
graph_sources.py
----------------
Provides two graph source types that plug into the same agent/visualizer
pipeline as the existing Grid class:

    CityGraph   — loads a real weighted graph from a coordinates CSV and
                  an adjacency list text file.  Edge weights are the
                  Haversine great-circle distance (km) between cities.

    RandomGraph — procedurally generated weighted graph parameterized by
                  number of nodes (N), expected branching factor (b),
                  edge weight distribution, and a random seed.

Both classes expose the same interface the rest of the codebase expects:

    .G          nx.Graph (undirected, weighted)
    .start      node ID of the chosen start
    .goal       node ID of the chosen goal
    .size       used by agents that read this attribute (set to N)
    .pos        dict  node_id → (x, y)  for visualization
    .node_label(node_id) → display string

Node selection is interactive: a matplotlib window renders the graph and
the user clicks two nodes (first = start, second = goal).

Classes
-------
CityGraph
RandomGraph

Helper functions (module-private)
----------------------------------
_haversine(lat1, lon1, lat2, lon2)   → distance in km
_pick_nodes_interactively(G, pos, labels, title)  → (start, goal)
"""

import math
import random
import csv
import re
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in kilometres between two (lat, lon) points.

    Uses the Haversine formula, which is accurate to within ~0.5% for
    distances up to a few thousand km.
    """
    R = 6371.0  # Earth mean radius, km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Interactive node picker
# ---------------------------------------------------------------------------

def _pick_nodes_interactively(G, pos, labels, title="Click start then goal"):
    """
    Render *G* in a matplotlib window and let the user click two nodes.

    The first click selects the start (gold), the second selects the goal
    (dodger blue).  The window closes automatically after both are chosen.

    Parameters
    ----------
    G      : nx.Graph
    pos    : dict  node → (x, y)
    labels : dict  node → display string
    title  : str

    Returns
    -------
    (start_id, goal_id)
    """
    chosen = []

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax.set_facecolor('#f8f8f8')

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.2, edge_color='#aaaaaa', alpha=0.7)

    # Draw edge weight labels
    edge_labels = {(u, v): f"{d['weight']:.0f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                  font_size=6, font_color='#555555', label_pos=0.35)

    # Draw nodes — larger so they're easier to click
    node_list = list(G.nodes())
    node_collection = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=node_list,
        node_color='lightgreen',
        node_size=600,
        edgecolors='#333333',
        linewidths=0.8
    )
    # Labels offset slightly above each node to avoid overlap
    label_pos = {n: (x, y + 0.45) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels=labels, ax=ax, font_size=7, font_weight='bold')

    # Scatter artists for start/goal highlights (updated on click)
    start_scatter = ax.scatter([], [], c='gold',       s=400, zorder=6, edgecolors='black', linewidths=1.2)
    goal_scatter  = ax.scatter([], [], c='dodgerblue', s=400, zorder=6, edgecolors='black', linewidths=1.2, marker='X')

    status = ax.text(0.5, -0.04, "Click a node to set START",
                     transform=ax.transAxes, ha='center', fontsize=11,
                     color='darkorange', fontweight='bold')

    legend_handles = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='City'),
        plt.scatter([], [], c='gold',       s=80, edgecolors='black', label='Start'),
        plt.scatter([], [], c='dodgerblue', s=80, edgecolors='black', marker='X', label='Goal'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9)
    ax.axis('off')
    plt.tight_layout()

    # Build a KD-tree-style lookup: map each node to its pixel coords for click detection
    node_xy = np.array([pos[n] for n in node_list])

    def _nearest_node(xdata, ydata):
        """Return the node ID closest to the click in data coordinates."""
        click = np.array([xdata, ydata])
        dists = np.linalg.norm(node_xy - click, axis=1)
        return node_list[int(np.argmin(dists))]

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        node = _nearest_node(event.xdata, event.ydata)
        if node in chosen:
            return  # don't pick the same node twice

        chosen.append(node)
        x, y = pos[node]

        if len(chosen) == 1:
            start_scatter.set_offsets([[x, y]])
            status.set_text(f"Start: {labels[node]}  —  now click a node to set GOAL")
            status.set_color('steelblue')
        elif len(chosen) == 2:
            goal_scatter.set_offsets([[x, y]])
            status.set_text(f"Start: {labels[chosen[0]]}   Goal: {labels[chosen[1]]}   — closing…")
            status.set_color('green')
            fig.canvas.draw_idle()
            # Short pause so the user sees the confirmation, then close
            timer = fig.canvas.new_timer(interval=900)
            timer.add_callback(plt.close, fig)
            timer.start()

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show(block=True)

    if len(chosen) < 2:
        raise RuntimeError("Node selection cancelled — fewer than 2 nodes were picked.")

    return chosen[0], chosen[1]


# ---------------------------------------------------------------------------
# CityGraph
# ---------------------------------------------------------------------------

class CityGraph:
    """
    Weighted undirected graph built from a coordinates CSV and an adjacency
    text file.

    The graph uses integer node IDs (0-based index into the sorted city list)
    so agents that expect integer IDs continue to work.  Human-readable city
    names are stored in self.labels.

    Edge weights are the Haversine great-circle distance (km) between cities,
    rounded to 1 decimal place.

    Parameters
    ----------
    coord_path : str
        Path to CSV file.  Each row: CityName, latitude, longitude
    adj_path : str
        Path to plain-text adjacency file.  Each non-empty line contains
        exactly two whitespace-separated city names that share an edge.
    interactive : bool, optional
        If True (default), open a matplotlib window for node selection.
        If False, start and goal are left as None (useful for benchmarking).

    Attributes
    ----------
    G       : nx.Graph    weighted undirected graph
    pos     : dict        node_id → (x, y)  longitude/latitude scaled for display
    labels  : dict        node_id → city name string
    name_to_id : dict     city name → node_id
    start   : int or None
    goal    : int or None
    size    : int         number of nodes (for agent compatibility)
    seed    : None        city graphs have no seed (deterministic)
    """

    seed = None

    def __init__(self, coord_path: str, adj_path: str, interactive: bool = True):
        self.G         = nx.Graph()
        self.labels    = {}
        self.name_to_id = {}
        self.start     = None
        self.goal      = None

        self._load_coords(coord_path)
        self._load_edges(adj_path)
        self.size = self.G.number_of_nodes()

        # Display layout: geographic (lon/lat), scaled to a wide canvas,
        # then lightly repulsed so no two labels collide.
        # Real lat/lon stay on node attributes for haversine; display pos is separate.
        lats = [self.G.nodes[n]['lat'] for n in self.G.nodes()]
        lons = [self.G.nodes[n]['lon'] for n in self.G.nodes()]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        span_lon = max_lon - min_lon or 1
        span_lat = max_lat - min_lat or 1

        # Wide aspect ratio (22 x 10) so the east-west spread of Kansas shows
        raw = {
            n: np.array([
                (self.G.nodes[n]['lon'] - min_lon) / span_lon * 22,
                (self.G.nodes[n]['lat'] - min_lat) / span_lat * 10,
            ])
            for n in self.G.nodes()
        }

        # Iterative repulsion: push nodes that are too close apart
        MIN_DIST = 1.2   # minimum display-unit separation between node centres
        for _ in range(300):
            moved = False
            nodes = list(raw.keys())
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    ni, nj = nodes[i], nodes[j]
                    delta = raw[ni] - raw[nj]
                    d = float(np.linalg.norm(delta))
                    if d < MIN_DIST and d > 1e-6:
                        push = (MIN_DIST - d) / 2 * delta / d
                        raw[ni] = raw[ni] + push
                        raw[nj] = raw[nj] - push
                        moved = True
            if not moved:
                break

        self.pos = {n: (float(raw[n][0]), float(raw[n][1])) for n in raw}

        if interactive:
            self.start, self.goal = _pick_nodes_interactively(
                self.G, self.pos, self.labels,
                title="Kansas Cities — click START then GOAL"
            )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_coords(self, path: str):
        """Parse CSV and add nodes to self.G."""
        with open(path, newline='') as f:
            reader = csv.reader(f)
            idx = 0
            for row in reader:
                row = [c.strip() for c in row]
                if not row or not row[0]:
                    continue
                name, lat, lon = row[0], float(row[1]), float(row[2])
                self.G.add_node(idx, name=name, lat=lat, lon=lon, passable=True)
                self.labels[idx]      = name.replace('_', ' ')
                self.name_to_id[name] = idx
                idx += 1

    def _load_edges(self, path: str):
        """Parse adjacency file and add weighted edges to self.G."""
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                a, b = parts[0], parts[1]
                if a not in self.name_to_id or b not in self.name_to_id:
                    # Unknown city name — skip silently
                    continue
                ia, ib = self.name_to_id[a], self.name_to_id[b]
                if self.G.has_edge(ia, ib):
                    continue  # already added (adjacency file may duplicate)
                na = self.G.nodes[ia]
                nb = self.G.nodes[ib]
                dist = round(_haversine(na['lat'], na['lon'], nb['lat'], nb['lon']), 1)
                self.G.add_edge(ia, ib, weight=dist)

    # ------------------------------------------------------------------
    # Interface helpers
    # ------------------------------------------------------------------

    def node_label(self, node_id: int) -> str:
        """Return the display label for a node."""
        return self.labels.get(node_id, str(node_id))

    def neighbors_clockwise(self, node_id: int):
        """
        Return neighbors sorted by edge weight (ascending).

        CityGraph has no directional geometry, so 'clockwise' is replaced
        by weight-ascending order — cheapest neighbor first.  Agents that
        call neighbors_clockwise() will still get a deterministic order.
        """
        return sorted(self.G.neighbors(node_id),
                      key=lambda n: self.G[node_id][n]['weight'])

    def edge_weight(self, u: int, v: int) -> float:
        """Return the weight of edge (u, v)."""
        return self.G[u][v]['weight']

    def visualize(self):
        """
        Render the city graph with start/goal highlighted.

        Called by main.py or standalone.  Does not block if start/goal
        are not yet set (shows the raw graph).
        """
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.set_facecolor('#f8f8f8')

        nx.draw_networkx_edges(self.G, self.pos, ax=ax, width=1.2,
                               edge_color='#aaaaaa', alpha=0.7)
        edge_labels = {(u, v): f"{d['weight']:.0f}"
                       for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels,
                                      ax=ax, font_size=6, font_color='#555555')

        passable = [n for n in self.G.nodes()
                    if n not in (self.start, self.goal)]
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=passable,
                               node_color='lightgreen', node_size=600,
                               edgecolors='black', linewidths=0.8)
        if self.start is not None:
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax,
                                   nodelist=[self.start], node_color='gold',
                                   node_size=700, edgecolors='black')
        if self.goal is not None:
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax,
                                   nodelist=[self.goal], node_color='dodgerblue',
                                   node_size=700, edgecolors='black', node_shape='X')

        label_pos = {n: (x, y + 0.45) for n, (x, y) in self.pos.items()}
        nx.draw_networkx_labels(self.G, label_pos, labels=self.labels,
                                ax=ax, font_size=7, font_weight='bold')
        ax.set_title(f"Kansas City Graph  ({self.G.number_of_nodes()} nodes, "
                     f"{self.G.number_of_edges()} edges)", fontsize=13)
        ax.axis('off')
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# RandomGraph
# ---------------------------------------------------------------------------

class RandomGraph:
    """
    Procedurally generated weighted undirected graph.

    Each node is placed uniformly at random in a [0, 100] × [0, 100]
    coordinate space.  Each node's degree is drawn from Poisson(b), giving
    natural degree variance around the expected branching factor.  The node
    is then connected to that many of its nearest unconnected neighbors.
    Edge weights are whole-integer Euclidean distances scaled by a per-edge
    multiplier drawn uniformly from weight_range, guaranteed ≥ 1.

    Connectivity is controlled by the connectedness parameter:
        1.0 — always fully connected (stitch all isolated components).
        0.0 — no stitching (graph may be disconnected).
        0.5 — stitch only if the largest component covers < 50% of nodes.
    In practice values ≥ 0.8 reliably give a single connected component.

    Parameters
    ----------
    n : int
        Number of nodes (default 20).
    b : int
        Expected branching factor — each node's actual degree is drawn from
        Poisson(b), then clipped to [1, n-1] (default 3).
    weight_range : tuple[int, int]
        (low, high) inclusive range for the integer weight multiplier applied
        per edge (default (1, 10)).
    connectedness : float  [0, 1]
        Fraction of nodes that must be in the largest component before
        stitching stops.  1.0 = always fully connected (default).
        Only has a visible effect when b is small relative to n (e.g. b=1
        or b=2 with n≥20); with b≥3 the Poisson wiring typically produces
        a connected graph on its own and stitching never runs.
    seed : int or None
        Random seed for reproducibility.  Recorded in self.seed.
    interactive : bool
        If True (default), open a matplotlib window for node selection.

    Attributes
    ----------
    G       : nx.Graph
    pos     : dict  node_id → (x, y)
    labels  : dict  node_id → str  (e.g. 'N0', 'N1', …)
    start   : int or None
    goal    : int or None
    size    : int   = n
    seed    : int
    b_actual : float  observed mean degree / 2  (reported in title/visualize)
    """

    def __init__(self,
                 n: int = 20,
                 b: int = 3,
                 weight_range: tuple = (1, 10),
                 connectedness: float = 1.0,
                 seed: int | None = None,
                 interactive: bool = True):

        if seed is None:
            seed = random.randint(0, 2**31)
        self.seed = seed
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        self.size   = n
        self.start  = None
        self.goal   = None
        self.G      = nx.Graph()
        self.labels = {i: f'N{i}' for i in range(n)}
        self._connectedness = connectedness

        # Place nodes at random positions
        xs = np_rng.uniform(0, 100, n)
        ys = np_rng.uniform(0, 100, n)
        self.pos = {i: (float(xs[i]), float(ys[i])) for i in range(n)}

        for i in range(n):
            self.G.add_node(i, x=float(xs[i]), y=float(ys[i]), passable=True)

        # Poisson-sampled degree wiring
        # Each node draws its target degree from Poisson(b), clipped to [1, n-1].
        # It then connects to that many nearest not-yet-connected neighbors.
        coords = np.stack([xs, ys], axis=1)
        for i in range(n):
            k = int(np.clip(np_rng.poisson(b), 1, n - 1))
            dists = np.linalg.norm(coords - coords[i], axis=1)
            dists[i] = np.inf
            nearest = np.argsort(dists)
            added = 0
            for j in nearest:
                if added >= k:
                    break
                j = int(j)
                raw_dist = float(dists[j])
                if raw_dist == float('inf'):
                    break  # all remaining neighbors are unreachable
                if not self.G.has_edge(i, j):
                    multiplier = rng.randint(*weight_range)
                    w = max(1, round(raw_dist * multiplier / 10))
                    self.G.add_edge(i, j, weight=w)
                    added += 1

        # Connectivity enforcement scaled by connectedness parameter
        self._stitch_components()

        # Record observed branching factor
        self.b_actual = round(self.G.number_of_edges() * 2 / n, 2)

        if interactive:
            self.start, self.goal = _pick_nodes_interactively(
                self.G, self.pos, self.labels,
                title=(f"Random Graph  "
                       f"(N={n}, b={b}, b_obs={self.b_actual}, "
                       f"conn={connectedness}, seed={seed})"
                       f"  — click START then GOAL")
            )

    # ------------------------------------------------------------------
    # Connectivity enforcement
    # ------------------------------------------------------------------

    def _stitch_components(self):
        """
        Stitch isolated components until the largest component contains at
        least (connectedness × n) nodes, or the graph is fully connected.

        connectedness=1.0 → always fully connected.
        connectedness=0.5 → stop once ≥50% of nodes share one component.
        connectedness=0.1 → stop once ≥10% share one component (very sparse).

        The loop always runs at least once so the initial wiring pass can
        produce multiple small components before we check the threshold.
        """
        target = self._connectedness * self.size
        coords = np.array([list(self.pos[i]) for i in range(self.size)])

        while True:
            components = sorted(nx.connected_components(self.G),
                                key=len, reverse=True)
            # Always stop if fully connected
            if len(components) == 1:
                break
            # Stop stitching once the largest component meets the target fraction
            if len(components[0]) >= target and self._connectedness < 1.0:
                break

            main  = list(components[0])
            other = list(components[1])
            main_coords  = coords[main]
            other_coords = coords[other]
            diffs = main_coords[:, None, :] - other_coords[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            mi, oi = np.unravel_index(np.argmin(dists), dists.shape)
            u, v = main[mi], other[oi]
            w = max(1, round(float(dists[mi, oi]) / 10))
            self.G.add_edge(u, v, weight=w)

    # ------------------------------------------------------------------
    # Interface helpers
    # ------------------------------------------------------------------

    def node_label(self, node_id: int) -> str:
        return self.labels.get(node_id, str(node_id))

    def neighbors_clockwise(self, node_id: int):
        """Return neighbors sorted by ascending edge weight."""
        return sorted(self.G.neighbors(node_id),
                      key=lambda n: self.G[node_id][n]['weight'])

    def edge_weight(self, u: int, v: int) -> float:
        return self.G[u][v]['weight']

    def visualize(self):
        """Render the random graph with start/goal highlighted."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor('#f8f8f8')

        nx.draw_networkx_edges(self.G, self.pos, ax=ax, width=1.2,
                               edge_color='#aaaaaa', alpha=0.7)
        edge_labels = {(u, v): f"{d['weight']:.1f}"
                       for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels,
                                      ax=ax, font_size=7, font_color='#555555')

        regular = [n for n in self.G.nodes()
                   if n not in (self.start, self.goal)]
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=regular,
                               node_color='lightgreen', node_size=400,
                               edgecolors='black')
        if self.start is not None:
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax,
                                   nodelist=[self.start], node_color='gold',
                                   node_size=500, edgecolors='black')
        if self.goal is not None:
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax,
                                   nodelist=[self.goal], node_color='dodgerblue',
                                   node_size=500, edgecolors='black')

        nx.draw_networkx_labels(self.G, self.pos, labels=self.labels,
                                ax=ax, font_size=9, font_weight='bold')
        ax.set_title(
            f"Random Graph  (N={self.size}, b_obs={self.b_actual}, "
            f"conn={self._connectedness}, seed={self.seed})",
            fontsize=13
        )
        ax.axis('off')
        plt.tight_layout()
        plt.show()