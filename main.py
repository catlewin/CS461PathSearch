from environment_generation import create_grid, generate_obstacles, get_start_and_goal, grid_to_graph
from agent import bfs, dfs
from search_visualization import visualize_side_by_side


# Create grid with obstacles, start & end
grid_dim = 8
grid = generate_obstacles(create_grid(grid_dim))
start, goal = get_start_and_goal(grid)

# Matrix --> NetworkX Graph
grid_graph = grid_to_graph(grid)

# Call BFS, return
#   boolean if path to goal found
#   list of tuples, order of discovery, then visiting
#   list of nodes, visiting order
#   list of nodes, path from start to goal, empty if no path
#   dictionary, children nodes: parent node
found, events, sequence, path, parent = bfs(grid_graph, start, goal, grid_dim)

# Call visualization animation
visualize_side_by_side(found, events, sequence, path, parent, grid_graph, start, goal, grid_dim, True)