from environment_generation import Grid
from agent import BFSAgent, DFSAgent, IDDFSAgent, GreedyAgent, AStarAgent
from search_visualization import Visualizer

import random

random.seed(42)

# Build grid environment (nodes, obstacles, edges, start & goal all set in __init__)
grid = Grid(size=12)

# BFS
# bfs_agent = BFSAgent(grid)
# bfs_agent.search()
# Visualizer(bfs_agent).show_all()

# DFS
# dfs_agent = DFSAgent(grid)
# dfs_agent.search()
# Visualizer(dfs_agent).show_all()

# IDDFS — use search_verbose() for visualization, search() for benchmarking
# iddfs_agent = IDDFSAgent(grid)
# iddfs_agent.search_verbose()
# Visualizer(iddfs_agent).show_all()

# Greedy best-first
greedy_agent = GreedyAgent(grid)
greedy_agent.search()
Visualizer(greedy_agent).show_all()

# # A*
# astar_agent = AStarAgent(grid)
# astar_agent.search()
# Visualizer(astar_agent).show_all()