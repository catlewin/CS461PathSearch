"""
metrics.py
----------
Collects and reports performance metrics for any search agent run.

Usage
-----
    from metrics import run_with_metrics
    result = run_with_metrics(agent)   # calls agent.search() internally
    print(result.summary())

Or for IDDFS verbose mode:
    result = run_with_metrics(agent, method='search_verbose')

Metrics collected
-----------------
Runtime
    Wall-clock seconds via time.perf_counter(), wrapping the search() call.

Memory — two measures, documented separately:
    algorithmic_peak_frontier : int
        Peak number of nodes simultaneously in the open list (frontier).
        Measured by replaying agent.events after search completes and
        tracking the high-water mark of the discovered-but-not-yet-visited
        set.  This is the algorithm's own memory footprint independent of
        Python overhead.

    algorithmic_explored : int
        Total nodes in the explored (closed) set at termination — i.e.
        nodes that were fully expanded.

    process_peak_kb : float
        Peak additional process memory (KB) consumed during search(),
        measured via tracemalloc.  Captures Python object allocation
        overhead on top of the algorithmic footprint.  Note: tracemalloc
        measures the Python heap only; OS-level RSS may differ.

Search effort
    nodes_generated : int
        Total nodes added to the frontier (discover/push events), including
        the start node.  Counts re-discoveries for algorithms that allow
        them (e.g. A* with cost updates).

    nodes_expanded : int
        Total nodes popped from the frontier and processed (visit events).

    branching_factor_avg : float
        Mean number of successors generated per expanded node.
        = nodes_generated / nodes_expanded  (excluding the goal expansion
        since search terminates immediately on goal pop).

    branching_factor_max : int
        Maximum number of successors generated from any single node.

    solution_depth : int
        Number of edges on the solution path (0 if no path found).

    path_cost : float
        Sum of edge weights along the solution path (0.0 if no path).
        For unit-cost graphs this equals solution_depth.

Optimality
    is_optimal : bool | None
        True  — algorithm guarantees optimal cost on this graph type.
        False — algorithm is not guaranteed optimal (Greedy, DFS, IDDFS
                on weighted graphs).
        None  — unknown / not applicable.
"""

import time
import tracemalloc
from collections import defaultdict


# ---------------------------------------------------------------------------
# Optimality table
# ---------------------------------------------------------------------------

# Maps agent class name → (optimal_on_unweighted, optimal_on_weighted)
_OPTIMALITY = {
    'BFSAgent':    (True,  False),   # optimal hops, not weighted cost
    'DFSAgent':    (False, False),
    'IDDFSAgent':  (True,  False),   # optimal hops on unweighted only
    'GreedyAgent': (False, False),
    'AStarAgent':  (True,  True),    # admissible heuristic → optimal
}


def _is_weighted(env) -> bool:
    """Return True if the environment uses non-unit edge weights."""
    for u, v, d in env.G.edges(data=True):
        if d.get('weight', 1) != 1:
            return True
    return False


def _optimality_label(agent) -> tuple[bool | None, str]:
    """
    Return (is_optimal, explanation) for the given agent + environment.
    """
    cls = type(agent).__name__
    weighted = _is_weighted(agent.grid)
    opt_unw, opt_w = _OPTIMALITY.get(cls, (None, None))

    if opt_unw is None:
        return None, 'unknown'

    if weighted:
        is_opt = opt_w
        reason = ('optimal (A* + admissible heuristic)' if is_opt
                  else f'not guaranteed optimal on weighted graphs ({cls})')
    else:
        is_opt = opt_unw
        reason = ('optimal (unweighted graph)' if is_opt
                  else f'not guaranteed optimal ({cls})')

    return is_opt, reason


# ---------------------------------------------------------------------------
# MetricsResult
# ---------------------------------------------------------------------------

class MetricsResult:
    """
    Holds all collected metrics for one agent run.

    Attributes mirror the module-level docstring fields.
    """

    def __init__(self):
        self.agent_name            = ''
        self.graph_name            = ''
        self.seed                  = None

        # Runtime
        self.runtime_s             = 0.0

        # Memory
        self.algorithmic_peak_frontier = 0
        self.algorithmic_explored      = 0
        self.process_peak_kb           = 0.0

        # Search effort
        self.nodes_generated       = 0
        self.nodes_expanded        = 0
        self.branching_factor_avg  = 0.0
        self.branching_factor_max  = 0
        self.solution_depth        = 0
        self.path_cost             = 0.0

        # Optimality
        self.is_optimal            = None
        self.optimality_reason     = ''

        # Path
        self.path                  = []

    def summary(self) -> str:
        """Return a compact multi-line summary string."""
        lines = [
            f"Agent          : {self.agent_name}",
            f"Graph          : {self.graph_name}",
            f"Seed           : {self.seed}",
            f"─" * 38,
            f"Runtime        : {self.runtime_s*1000:.2f} ms",
            f"Process mem    : {self.process_peak_kb:.1f} KB peak",
            f"─" * 38,
            f"Nodes generated: {self.nodes_generated}",
            f"Nodes expanded : {self.nodes_expanded}",
            f"Branch avg/max : {self.branching_factor_avg:.2f} / {self.branching_factor_max}",
            f"Solution depth : {self.solution_depth}",
            f"Path cost      : {self.path_cost:.2f}",
            f"─" * 38,
            f"Optimal        : {'✓' if self.is_optimal else ('✗' if self.is_optimal is False else '?')}  {self.optimality_reason}",
        ]
        return '\n'.join(lines)

    def as_dict(self) -> dict:
        """Return metrics as a flat dict (for batch comparison tables)."""
        return {
            'agent':            self.agent_name,
            'graph':            self.graph_name,
            'seed':             self.seed,
            'runtime_ms':       round(self.runtime_s * 1000, 3),
            'process_kb':       round(self.process_peak_kb, 2),
            'alg_frontier_peak': self.algorithmic_peak_frontier,
            'alg_explored':     self.algorithmic_explored,
            'nodes_generated':  self.nodes_generated,
            'nodes_expanded':   self.nodes_expanded,
            'branch_avg':       round(self.branching_factor_avg, 3),
            'branch_max':       self.branching_factor_max,
            'solution_depth':   self.solution_depth,
            'path_cost':        round(self.path_cost, 3),
            'is_optimal':       self.is_optimal,
        }


# ---------------------------------------------------------------------------
# Core collection function
# ---------------------------------------------------------------------------

def run_with_metrics(agent, method: str = 'search') -> MetricsResult:
    """
    Run agent.search() (or agent.<method>()) and collect all metrics.

    Parameters
    ----------
    agent  : any Agent subclass (BFSAgent, AStarAgent, …)
    method : str  name of the search method to call (default 'search').
             Use 'search_verbose' for IDDFSAgent visualization.

    Returns
    -------
    MetricsResult
    """
    result = MetricsResult()
    env    = agent.grid

    result.agent_name = type(agent).__name__
    result.graph_name = type(env).__name__
    result.seed       = getattr(env, 'seed', None)

    # ---- Runtime + process memory ----------------------------------------
    tracemalloc.start()
    t0 = time.perf_counter()
    getattr(agent, method)()
    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result.runtime_s      = t1 - t0
    result.process_peak_kb = peak_bytes / 1024

    # ---- Path ---------------------------------------------------------------
    result.path         = agent.reconstruct_path()
    result.solution_depth = len(result.path) - 1 if result.path else 0
    result.path_cost    = (
        sum(env.G[result.path[i]][result.path[i+1]].get('weight', 1)
            for i in range(len(result.path) - 1))
        if len(result.path) > 1 else 0.0
    )

    # ---- Search effort from events ----------------------------------------
    events = agent.events

    # nodes_generated: discover/push events (start is the first discover)
    generated_nodes = []
    for ev_type, payload in events:
        if ev_type in ('discover', 'push'):
            nid = payload[0] if isinstance(payload, tuple) else payload
            generated_nodes.append(nid)
    result.nodes_generated = len(generated_nodes)

    # nodes_expanded: visit events (unique node IDs popped and processed)
    expanded = [n for ev_type, n in events
                if ev_type == 'visit' and isinstance(n, int)]
    result.nodes_expanded = len(expanded)

    # Branching factor: successors generated per expanded node
    # Build a map of how many discover events follow each visit event
    successors_per_expansion = defaultdict(int)
    current_parent = None
    for ev_type, payload in events:
        if ev_type == 'visit' and isinstance(payload, int):
            current_parent = payload
        elif ev_type in ('discover', 'push') and current_parent is not None:
            successors_per_expansion[current_parent] += 1

    if successors_per_expansion:
        counts = list(successors_per_expansion.values())
        result.branching_factor_avg = sum(counts) / len(counts)
        result.branching_factor_max = max(counts)
    else:
        result.branching_factor_avg = 0.0
        result.branching_factor_max = 0

    # ---- Algorithmic memory: replay frontier size -------------------------
    frontier = set()
    explored = set()
    peak_frontier = 0

    for ev_type, payload in events:
        if ev_type in ('discover', 'push'):
            nid = payload[0] if isinstance(payload, tuple) else payload
            frontier.add(nid)
        elif ev_type == 'visit' and isinstance(payload, int):
            frontier.discard(payload)
            explored.add(payload)
        elif ev_type == 'new_iteration':
            # IDDFS restarts — reset frontier, keep explored count cumulative
            frontier.clear()

        peak_frontier = max(peak_frontier, len(frontier))

    result.algorithmic_peak_frontier = peak_frontier
    result.algorithmic_explored      = len(explored)

    # ---- Optimality -------------------------------------------------------
    result.is_optimal, result.optimality_reason = _optimality_label(agent)

    return result