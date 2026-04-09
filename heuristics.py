"""
heuristics.py
-------------
Heuristic functions for use with informed search agents (Greedy, A*).

Grid heuristics (for Grid environments)
----------------------------------------
manhattan(a, b, grid)
    Sum of horizontal + vertical steps.  Admissible and consistent on
    4-connected unit-cost grids.  NOT admissible on 8-connected grids
    (can overestimate diagonal moves).

chebyshev(a, b, grid)
    max(|Δrow|, |Δcol|).  Admissible and consistent on 8-connected unit-
    cost grids.  Overestimates on 4-connected grids.

euclidean_grid(a, b, grid)
    Straight-line distance in cell units.  Admissible on both 4- and
    8-connected grids (always ≤ true path length).  Slightly less informed
    than Manhattan on 4-connected grids.

Admissibility verification
--------------------------
A heuristic h is admissible iff h(n) ≤ h*(n) for all n, where h*(n) is
the true optimal cost from n to the goal.

    manhattan  on 4-connected unit grids:
        Each step changes row or col by 1, so the minimum steps ≥
        |Δrow| + |Δcol|.  Equality holds when no obstacles block the
        L-shaped direct path.  ∴ admissible. ✓

    chebyshev  on 8-connected unit grids:
        Each step can reduce max(|Δrow|, |Δcol|) by at most 1 (diagonal
        moves reduce both).  ∴ h*(n) ≥ max(|Δrow|, |Δcol|).  Admissible. ✓

    euclidean_grid on 4-/8-connected unit grids:
        sqrt(Δrow²+Δcol²) ≤ |Δrow|+|Δcol| (triangle inequality) and
        ≤ chebyshev-based bound, so h*(n) ≥ euclidean always.  Admissible. ✓

Graph heuristics (for CityGraph / RandomGraph environments)
------------------------------------------------------------
euclidean_graph(a, b, graph)
    Straight-line Euclidean distance in the graph's coordinate space.
    Admissible if edge weights reflect actual distances (never underestimates
    the true path cost).

haversine_graph(a, b, graph)
    Great-circle distance in km between two nodes that carry 'lat'/'lon'
    node attributes.  Intended for CityGraph; uses the same formula as
    edge weights so it is admissible by the triangle inequality. ✓

zero(a, b, *args, **kwargs)
    Always returns 0.  Reduces A* to Dijkstra's algorithm.  Useful as a
    baseline/sanity check.

Usage
-----
Pass a heuristic as a callable to GreedyAgent / AStarAgent:

    from heuristics import manhattan, chebyshev
    agent = AStarAgent(grid, heuristic=manhattan)
    agent = AStarAgent(grid, heuristic=chebyshev)

Each heuristic signature:

    h(node_a: int, node_b: int, env) -> float

where env is a Grid, CityGraph, or RandomGraph instance.
"""

import math


# ---------------------------------------------------------------------------
# Grid heuristics
# ---------------------------------------------------------------------------

def manhattan(a: int, b: int, env) -> float:
    """
    Manhattan distance between flat node IDs on a Grid.

    Admissible on 4-connected unit-cost grids. ✓
    May overestimate on 8-connected or weighted grids. ✗

    Parameters
    ----------
    a, b : int   flat node IDs  (row * size + col)
    env  : Grid
    """
    size = env.size
    ar, ac = divmod(a, size)
    br, bc = divmod(b, size)
    return float(abs(ar - br) + abs(ac - bc))


def chebyshev(a: int, b: int, env) -> float:
    """
    Chebyshev (chessboard) distance between flat node IDs on a Grid.

    Admissible on 8-connected unit-cost grids. ✓
    May underestimate on 4-connected grids (consistent but less informed).

    Parameters
    ----------
    a, b : int   flat node IDs
    env  : Grid
    """
    size = env.size
    ar, ac = divmod(a, size)
    br, bc = divmod(b, size)
    return float(max(abs(ar - br), abs(ac - bc)))


def euclidean_grid(a: int, b: int, env) -> float:
    """
    Euclidean straight-line distance between flat node IDs on a Grid.

    Admissible on both 4- and 8-connected unit-cost grids. ✓
    Less informed than Manhattan on 4-connected grids (lower bound is
    looser), but always admissible.

    Parameters
    ----------
    a, b : int   flat node IDs
    env  : Grid
    """
    size = env.size
    ar, ac = divmod(a, size)
    br, bc = divmod(b, size)
    return math.hypot(ar - br, ac - bc)


# ---------------------------------------------------------------------------
# Graph heuristics (CityGraph / RandomGraph)
# ---------------------------------------------------------------------------

def euclidean_graph(a: int, b: int, env) -> float:
    """
    Euclidean distance in the graph's 2D coordinate space.

    For RandomGraph: coordinates are the random (x, y) positions.
    For CityGraph: coordinates are the scaled (lon, lat) display positions.

    Admissible when edge weights are proportional to Euclidean distances
    (as in RandomGraph with weight_range=(1,1)).  For CityGraph use
    haversine_graph instead.

    Parameters
    ----------
    a, b : int       node IDs
    env  : CityGraph | RandomGraph
    """
    xa, ya = env.pos[a]
    xb, yb = env.pos[b]
    return math.hypot(xa - xb, ya - yb)


def haversine_graph(a: int, b: int, env) -> float:
    """
    Great-circle (Haversine) distance in km between two city nodes.

    Requires nodes to have 'lat' and 'lon' attributes (CityGraph).
    Edge weights in CityGraph are also Haversine distances, so this
    heuristic is admissible by the triangle inequality. ✓

    Parameters
    ----------
    a, b : int       node IDs
    env  : CityGraph
    """
    na = env.G.nodes[a]
    nb = env.G.nodes[b]
    R = 6371.0
    phi1, phi2 = math.radians(na['lat']), math.radians(nb['lat'])
    dphi = math.radians(nb['lat'] - na['lat'])
    dlam = math.radians(nb['lon'] - na['lon'])
    a_val = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a_val), math.sqrt(1 - a_val))


def zero(a: int, b: int, env) -> float:
    """
    Null heuristic — always returns 0.

    Reduces A* to Dijkstra's algorithm.  Admissible trivially. ✓
    Useful as a baseline or for benchmarking the overhead of heuristic
    computation vs. pure uniform-cost search.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Heuristic registry — maps display name → callable
# ---------------------------------------------------------------------------

GRID_HEURISTICS = {
    'Manhattan':      manhattan,
    'Chebyshev':      chebyshev,
    'Euclidean':      euclidean_grid,
    'Zero (Dijkstra)': zero,
}

GRAPH_HEURISTICS = {
    'Euclidean':      euclidean_graph,
    'Haversine':      haversine_graph,
    'Zero (Dijkstra)': zero,
}