import random

from environment_generation import Grid
from agent import BFSAgent, DFSAgent
from search_visualization import Visualizer

random.seed(42)

# Build grid environment (nodes, obstacles, edges, start & goal all set in __init__)
grid = Grid(size=5)

# BFS agent
bfs_agent = BFSAgent(grid)
bfs_agent.search()
bfs_vis = Visualizer(bfs_agent)
# bfs_vis.show_grid()
# bfs_vis.show_tree()
bfs_vis.show_all()

# # DFS agent
dfs_agent = DFSAgent(grid)
dfs_agent.search()
dfs_vis = Visualizer(dfs_agent)
# dfs_vis.show_grid()
# dfs_vis.show_tree()
dfs_vis.show_all()