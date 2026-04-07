"""
agent.py
--------
Defines a base Agent class and BFSAgent / DFSAgent subclasses for
grid-based pathfinding. Each agent holds a Grid instance and runs its
search algorithm against grid.G, grid.start, and grid.goal.

All search state is stored on the agent after search() is called:
    - parent : dict  — maps each discovered node to its parent (used for
                       path reconstruction and tree visualization)
    - events : list  — ordered ('discover', node_id) and ('visit', node_id)
                       tuples recording the full traversal history

Everything else is derived on demand:
    - path             via agent.reconstruct_path()
    - visitation order via agent.visit_sequence()
    - found            via agent.found  (property)

Classes
-------
Agent
    Abstract base class. Owns a Grid, parent dict, and events list.

BFSAgent(Agent)
    Breadth-first search implementation.

DFSAgent(Agent)
    Depth-first search implementation.
"""

from abc import ABC, abstractmethod
from collections import deque
from environment_generation import Grid


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
        Ordered sequence of ('discover', node_id) and ('visit', node_id)
        events recorded during search(). Empty until search() is called.
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        self.parent: dict = {}
        self.events: list = []

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

        Derived by filtering 'visit' events — no separate list needed.

        Returns
        -------
        list[int]
            Node IDs in visitation order.
        """
        return [node for event, node in self.events if event == 'visit']


class BFSAgent(Agent):
    """
    Breadth-first search agent.

    Explores the grid level by level from grid.start toward grid.goal.
    Nodes are marked 'discover' when enqueued and 'visit' when dequeued.
    Neighbors are always processed clockwise starting from right, so the
    discovered frontier at each depth level is deterministically ordered.
    """

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

    def search(self):
        """
        Run DFS on self.grid. Populates self.parent and self.events.

        Each stack entry is (node_id, path_to_node) so that when a node is
        popped, the exact path that led to it is known immediately — no
        parent-chain reconstruction needed.

        Event order:
            ('visit', node_id) — emitted when node is popped and processed
            for the first time.
        """
        self.parent = {self.grid.start: None}
        self.events = []

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
                    stack.append((neighbor, path + [neighbor]))

        self._path = []

    def reconstruct_path(self) -> list[int]:
        """Return the path recorded during search, or [] if not found."""
        return getattr(self, '_path', [])