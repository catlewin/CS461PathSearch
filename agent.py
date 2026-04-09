"""
agent.py
--------
Defines a base Agent class and search algorithm subclasses for grid-based
pathfinding. Each agent holds a Grid instance and runs its search algorithm
against grid.G, grid.start, and grid.goal.

All search state is stored on the agent after search() is called:
    - parent : dict  — maps each discovered node to its parent (used for
                       path reconstruction and tree visualization)
    - events : list  — ordered ('discover', node_id), ('visit', node_id),
                       and ('new_iteration', depth) tuples recording the
                       full traversal history

Everything else is derived on demand:
    - path             via agent.reconstruct_path()
    - visitation order via agent.visit_sequence()
    - found            via agent.found  (property)

Classes
-------
Agent
    Abstract base class. Owns a Grid, parent dict, and events list.

BFSAgent(Agent)
    Breadth-first search — optimal on unweighted grids.

DFSAgent(Agent)
    Depth-first search — follows one branch to completion before backtracking.

IDDFSAgent(Agent)
    Iterative deepening DFS — DFS memory footprint with BFS optimal path length.
    search()         : pure implementation, records only the final iteration.
    search_verbose() : records all iterations for visualization, emitting
                       ('new_iteration', depth) events at each restart.

GreedyAgent(Agent)
    Best-first (greedy) search — expands lowest h(n) = Manhattan distance to goal.
    Fast but not guaranteed optimal.

AStarAgent(Agent)
    A* search — expands lowest f(n) = g(n) + h(n). Optimal with admissible
    Manhattan distance heuristic on this uniform-cost grid.

Module helpers
--------------
_manhattan(a, b, size)
    Manhattan distance between two flat node IDs on a grid of given size.
"""

from abc import ABC, abstractmethod
from collections import deque
import heapq
from environment_generation import Grid
from heuristics import manhattan as _default_manhattan


def _manhattan(a: int, b: int, size: int) -> int:
    """
    Manhattan distance between flat node IDs a and b on a (size x size) grid.
    Kept for internal backward-compat; agents now accept a heuristic callable.
    """
    ar, ac = divmod(a, size)
    br, bc = divmod(b, size)
    return abs(ar - br) + abs(ac - bc)


def _edge_weight(env, u: int, v: int) -> float:
    """
    Return the edge weight between u and v in any environment type.

    Grid environments use unit cost (1) unless edges carry a 'weight'
    attribute.  CityGraph and RandomGraph always carry weights.
    """
    if hasattr(env, 'edge_weight'):
        return env.edge_weight(u, v)
    data = env.G[u][v]
    return float(data.get('weight', 1))


class Agent(ABC):
    """
    Abstract base class for grid search agents.

    Parameters
    ----------
    grid : Grid
        The environment to search. Provides grid.G, grid.start, grid.goal.

    Attributes
    ----------
    grid : Grid
    parent : dict
        Maps each discovered node ID to its parent node ID.
        Populated by search(). Empty until search() is called.
    events : list[tuple]
        Ordered sequence of ('discover', node_id), ('visit', node_id), and
        ('new_iteration', depth) events recorded during search().
        Empty until search() is called.
    shows_discovered : bool
        Class-level flag. True if this agent emits meaningful 'discover'
        events (frontier visible before visitation — BFS-like). False for
        agents where discover and visit are effectively simultaneous (DFS-like).
        Used by Visualizer to enable/disable the yellow discovered state.
    """

    shows_discovered: bool = False

    def __init__(self, grid, heuristic=None):
        """
        Parameters
        ----------
        grid      : Grid | CityGraph | RandomGraph
        heuristic : callable(a, b, env) -> float, optional
            Injected heuristic.  None means use the agent-level default.
        """
        self.grid = grid
        self.heuristic = heuristic
        self.parent: dict = {}
        self.events: list = []
        self.visit_order: dict = {}

    def _h(self, a: int, b: int) -> float:
        """Evaluate heuristic h(a -> goal) using injected fn or default."""
        if self.heuristic is not None:
            return float(self.heuristic(a, b, self.grid))
        env = self.grid
        if hasattr(env, 'pos'):
            if 'lat' in env.G.nodes[a]:
                from heuristics import haversine_graph
                return haversine_graph(a, b, env)
            from heuristics import euclidean_graph
            return euclidean_graph(a, b, env)
        return _default_manhattan(a, b, env)

    @abstractmethod
    def search(self):
        """Run the search algorithm. Populates self.parent and self.events."""
        pass

    @property
    def found(self) -> bool:
        """True if the goal was reached during the last search() call."""
        return self.grid.goal in self.parent

    def reconstruct_path(self) -> list[int]:
        """
        Trace self.parent from goal back to start.

        Returns
        -------
        list[int]
            Node IDs from start to goal inclusive, or [] if not found.
        """
        if not self.found:
            return []

        path, node = [], self.grid.goal
        while node is not None:
            path.append(node)
            node = self.parent[node]
        path.reverse()
        return path if path[0] == self.grid.start else []

    def visit_sequence(self) -> list[int]:
        """
        Return nodes in the order they were visited (popped & processed).

        Also rebuilds self.visit_order as a node_id → 1-based index dict.

        Returns
        -------
        list[int]
            Node IDs in visitation order.
        """
        seq = [node for event, node in self.events
               if event == 'visit' and isinstance(node, int)]
        self.visit_order = {node: i + 1 for i, node in enumerate(seq)}
        return seq


class BFSAgent(Agent):
    """
    Breadth-first search agent.

    Explores the grid level by level from grid.start toward grid.goal.
    Nodes are marked 'discover' when enqueued and 'visit' when dequeued.
    Neighbors are always processed clockwise starting from right, so the
    discovered frontier at each depth level is deterministically ordered.
    """

    shows_discovered = True

    def search(self):
        """
        Run BFS on self.grid. Populates self.parent and self.events.

        Event order per node:
            1. ('discover', node)   — emitted when node is first enqueued
            2. ('visit',    node)   — emitted when node is dequeued & processed
            3. ('discover', n1),
               ('discover', n2) ... — each unvisited neighbor discovered clockwise

        All neighbors of a fully-visited depth level are discovered before
        any of them are visited (standard BFS frontier behaviour).
        """
        self.parent = {self.grid.start: None}
        self.events = [('discover', self.grid.start)]
        queue = deque([self.grid.start])

        while queue:
            curr = queue.popleft()
            self.events.append(('visit', curr))

            if curr == self.grid.goal:
                return

            for neighbor in self.grid.neighbors_clockwise(curr):
                if neighbor not in self.parent:
                    self.parent[neighbor] = curr
                    self.events.append(('discover', neighbor))
                    queue.append(neighbor)


class DFSAgent(Agent):
    """
    Depth-first search agent.

    Explores the grid by following each branch as deep as possible before
    backtracking. Traversal priority is clockwise from right: the agent
    always tries to go right first, then down, left, up.

    DFS has no meaningful 'discover' frontier state — a node is either
    unvisited or being visited — so only 'visit' events are emitted.

    Path is tracked explicitly via a path stack that mirrors the active
    branch, rather than reconstructed from parent after the fact. This
    avoids corruption from parent entries being overwritten across branches.
    """

    shows_discovered = False

    def search(self):
        """
        Run DFS on self.grid. Populates self.parent and self.events.

        Each stack entry is (node_id, path_to_node) so that when a node is
        popped, the exact path that led to it is known immediately — no
        parent-chain reconstruction needed.

        Event order:
            ('push',  node_id) — emitted when node is pushed onto the stack
            ('visit', node_id) — emitted when node is popped and processed
                                 for the first time
        """
        self.parent = {self.grid.start: None}
        self.events = [('push', (self.grid.start, None))]

        # Stack entries: (node_id, path_from_start_to_node)
        stack = [(self.grid.start, [self.grid.start])]
        visited = set()

        while stack:
            curr, path = stack.pop()

            if curr in visited:
                continue

            visited.add(curr)
            self.events.append(('visit', curr))

            if curr == self.grid.goal:
                # Store the valid path directly — no reconstruction needed
                self._path = path
                return

            # Push in reverse clockwise so right neighbor is on top of stack
            for neighbor in reversed(self.grid.neighbors_clockwise(curr)):
                if neighbor not in visited:
                    self.parent[neighbor] = curr
                    self.events.append(('push', (neighbor, curr)))
                    stack.append((neighbor, path + [neighbor]))

        self._path = []

    def reconstruct_path(self) -> list[int]:
        """Return the path recorded during search, or [] if not found."""
        return getattr(self, '_path', [])


# ---------------------------------------------------------------------------
# Iterative Deepening DFS
# ---------------------------------------------------------------------------

class IDDFSAgent(Agent):
    """
    Iterative deepening depth-first search agent.

    Runs depth-limited DFS repeatedly with increasing depth limits (0, 1, 2,
    ...) until the goal is found or all nodes are exhausted. Combines DFS's
    O(bd) memory footprint with BFS's optimal path length on unweighted grids.

    Two search modes:
        search()         — pure implementation. Only the current iteration's
                           stack is held in memory at any time. Records events
                           from the final (successful) iteration only. Use this
                           for benchmarking — memory and runtime reflect the
                           true algorithm cost.
        search_verbose() — records events from every iteration, emitting a
                           ('new_iteration', depth) marker between restarts.
                           Use this for visualization so each deepening step
                           is visible.

    Path is tracked explicitly per branch (same approach as DFSAgent) so
    parent-map corruption across iterations is not possible.
    """

    shows_discovered = False

    # ------------------------------------------------------------------
    # Core depth-limited search (shared by both modes)
    # ------------------------------------------------------------------

    def _depth_limited_search(self, depth_limit: int, verbose: bool) -> bool:
        """
        Run one depth-limited DFS pass from grid.start up to depth_limit.

        Populates self.parent and appends to self.events if verbose=True
        (or replaces self.events if verbose=False, keeping only this pass).
        Sets self._path on success.

        Parameters
        ----------
        depth_limit : int
        verbose : bool
            If True, appends events to self.events (caller prepended the
            'new_iteration' marker). If False, replaces self.events.

        Returns
        -------
        bool
            True if goal was reached within depth_limit.
        """
        if not verbose:
            self.events = []

        self.parent = {self.grid.start: None}

        # Stack entries: (node_id, path_to_node, depth)
        stack = [(self.grid.start, [self.grid.start], 0)]
        visited = set()
        self.events.append(('push', (self.grid.start, None)))

        while stack:
            curr, path, depth = stack.pop()

            if curr in visited:
                continue

            visited.add(curr)
            self.events.append(('visit', curr))

            if curr == self.grid.goal:
                self._path = path
                return True

            if depth >= depth_limit:
                continue  # at limit — don't expand further

            for neighbor in reversed(self.grid.neighbors_clockwise(curr)):
                if neighbor not in visited:
                    self.parent[neighbor] = curr
                    self.events.append(('push', (neighbor, curr)))
                    stack.append((neighbor, path + [neighbor], depth + 1))

        self._path = []
        return False

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    def search(self):
        """
        Pure IDDFS — benchmarking mode.

        Only the current iteration is held in memory. Events reflect the
        final successful iteration only (or the last exhausted iteration
        if no path exists).
        """
        self._path = []
        for depth_limit in range(self.grid.size ** 2):
            if self._depth_limited_search(depth_limit, verbose=False):
                return

    def search_verbose(self):
        """
        IDDFS — visualization mode.

        Accumulates events across all iterations. Each new iteration is
        prefixed with a ('new_iteration', depth_limit) event so the
        visualizer can mark restarts. At the end of each iteration a
        ('parent_snapshot', dict) event is emitted so the visualizer
        can reconstruct the correct tree for that iteration.
        All iteration events are retained in self.events.
        """
        self.events = []
        self._path = []
        for depth_limit in range(self.grid.size ** 2):
            self.events.append(('new_iteration', depth_limit))
            if self._depth_limited_search(depth_limit, verbose=True):
                self.events.append(('parent_snapshot', dict(self.parent)))
                return
            self.events.append(('parent_snapshot', dict(self.parent)))

    def reconstruct_path(self) -> list[int]:
        """Return the branch path recorded during search, or [] if not found."""
        return getattr(self, '_path', [])


# ---------------------------------------------------------------------------
# Best-First (Greedy) Search
# ---------------------------------------------------------------------------

class GreedyAgent(Agent):
    """
    Greedy best-first search agent.

    Expands the node with the lowest heuristic h(n) = Manhattan distance to
    goal, ignoring the cost already paid to reach n. Fast but not guaranteed
    to find the optimal path.

    Uses a min-heap priority queue. Heap entries: (h, node_id). node_id is
    used as tiebreaker to keep heap comparisons stable.

    Nodes are marked 'discover' when pushed and 'visit' when popped.
    """

    shows_discovered = True

    def search(self):
        """
        Run greedy best-first search. Populates self.parent and self.events.

        Also populates self.h_cost: dict mapping node_id → h(n) for every
        discovered node. Used by the visualizer for per-node hover info.

        Events emitted
        --------------
        ('discover', node_id) : node pushed onto the heap for the first time
        ('visit',    node_id) : node popped and processed
        """
        self.parent  = {self.grid.start: None}
        self.h_cost  = {}
        self.events  = [('discover', self.grid.start)]

        h_start = self._h(self.grid.start, self.grid.goal)
        self.h_cost[self.grid.start] = h_start
        heap    = [(h_start, self.grid.start)]
        visited = set()

        while heap:
            _, curr = heapq.heappop(heap)

            if curr in visited:
                continue

            visited.add(curr)
            self.events.append(('visit', curr))

            if curr == self.grid.goal:
                return

            for neighbor in self.grid.neighbors_clockwise(curr):
                if neighbor not in visited and neighbor not in self.parent:
                    self.parent[neighbor] = curr
                    h = self._h(neighbor, self.grid.goal)
                    self.h_cost[neighbor] = h
                    self.events.append(('discover', neighbor))
                    heapq.heappush(heap, (h, neighbor))


# ---------------------------------------------------------------------------
# A* Search
# ---------------------------------------------------------------------------

class AStarAgent(Agent):
    """
    A* search agent.

    Expands the node with the lowest f(n) = g(n) + h(n), where:
        g(n) — exact cost from start to n (uniform = number of steps)
        h(n) — Manhattan distance from n to goal (admissible heuristic)

    Guaranteed optimal on this uniform-cost grid because the heuristic
    never overestimates the true cost.

    A node may be pushed to the heap multiple times if a cheaper path to it
    is found. Stale heap entries are skipped on pop via a visited set.
    Parent and g_cost are updated whenever a cheaper path is found.

    Heap entries: (f, node_id). node_id breaks ties to keep comparisons stable.
    """

    shows_discovered = True

    def search(self):
        """
        Run A* search. Populates self.parent and self.events.

        Also populates self.g_cost and self.h_cost: dicts mapping node_id →
        g(n) and h(n) for every discovered node. Used by the visualizer for
        per-node hover info. g_cost is updated if a cheaper path is found.

        Events emitted
        --------------
        ('discover', node_id) : node pushed for the first time, or re-pushed
                                with a lower f score (cheaper path found)
        ('visit',    node_id) : node popped and committed as optimal
        """
        self.parent = {self.grid.start: None}
        self.g_cost = {self.grid.start: 0.0}
        self.h_cost = {}
        self.events = [('discover', self.grid.start)]

        h_start = self._h(self.grid.start, self.grid.goal)
        self.h_cost[self.grid.start] = h_start
        heap    = [(h_start, self.grid.start)]
        visited = set()

        while heap:
            _, curr = heapq.heappop(heap)

            if curr in visited:
                continue

            visited.add(curr)
            self.events.append(('visit', curr))

            if curr == self.grid.goal:
                return

            for neighbor in self.grid.neighbors_clockwise(curr):
                if neighbor in visited:
                    continue

                step_cost = _edge_weight(self.grid, curr, neighbor)
                tentative_g = self.g_cost[curr] + step_cost

                if tentative_g < self.g_cost.get(neighbor, float('inf')):
                    self.parent[neighbor] = curr
                    self.g_cost[neighbor] = tentative_g
                    h = self._h(neighbor, self.grid.goal)
                    self.h_cost[neighbor] = h
                    self.events.append(('discover', neighbor))
                    heapq.heappush(heap, (tentative_g + h, neighbor))