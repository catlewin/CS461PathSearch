from environment_generation import create_grid, generate_obstacles, get_start_and_goal, grid_to_graph
from agent import bfs, dfs
from search_visualization import visualize_side_by_side


# Create grid with obstacles, start & end
grid_dim = 8
grid = generate_obstacles(create_grid(grid_dim))
start, goal = get_start_and_goal(grid)

# Matrix --> NetworkX Graph
grid_graph = grid_to_graph(grid)

# BFS agent
found, events, sequence, path, parent = bfs(grid_graph, start, goal, grid_dim)

# Call visualization animation for BFS
visualize_side_by_side(found, events, sequence, path, parent, grid_graph, start, goal, grid_dim, True)

# DFS agent
found, sequence, path, parent = dfs(grid_graph, start, goal, grid_dim)

# Call visualization animation for DFS
visualize_side_by_side(found, sequence, sequence, path, parent, grid_graph, start, goal, grid_dim)