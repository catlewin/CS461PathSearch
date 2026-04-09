"""
main.py
-------
Two top-level modes:

    RUN_MODE = 'single'  -- run one agent, animate, show metrics panel
    RUN_MODE = 'batch'   -- benchmark all agents across complexity settings

Within single mode, set GRAPH to 'city', 'random', or 'grid'.
Within batch mode, set BATCH_GRAPH to 'city' or 'random'.
"""

import random
from environment_generation import Grid
from graph_sources import CityGraph, RandomGraph
from agent import BFSAgent, DFSAgent, IDDFSAgent, GreedyAgent, AStarAgent
from search_visualization import Visualizer
from heuristics import manhattan, haversine_graph, euclidean_graph
from metrics import run_with_metrics
from benchmarking import (
    single_run, batch_suite,
    RAND_AGENT_FACTORIES,
)

# ==========================================================================
# TOP-LEVEL MODE
# ==========================================================================
RUN_MODE   = 'single'   # 'single' | 'batch'
GRAPH      = 'city'     # for single: 'city' | 'random' | 'grid'

# ==========================================================================
# SINGLE RUN
# ==========================================================================
if RUN_MODE == 'single':

    if GRAPH == 'city':
        graph = CityGraph('coordinates.csv', 'Adjacencies.txt', interactive=True)
        agent = AStarAgent(graph, heuristic=haversine_graph)
        # agent = GreedyAgent(graph, heuristic=haversine_graph)
        # agent = BFSAgent(graph)
        single_run(agent, visualize=True)

    elif GRAPH == 'random':
        graph = RandomGraph(n=25, b=3, weight_range=(1, 10), seed=42, interactive=True)
        agent = AStarAgent(graph, heuristic=euclidean_graph)
        single_run(agent, visualize=True)

    elif GRAPH == 'grid':
        random.seed(42)
        grid = Grid(size=12)
        agent = AStarAgent(grid, heuristic=manhattan)
        single_run(agent, visualize=True)

# ==========================================================================
# BATCH COMPARE  (random graphs only)
# Batch benchmarking only applies to RandomGraph — the city graph is
# deterministic (fixed nodes, fixed weights) so running it with different
# seeds produces identical results.  Varying the seed only changes the
# random graph structure, giving meaningful mean ± std statistics.
#
# 3 complexity settings vary the Poisson branching factor b.
# Note: b_obs ≈ 2×b because each undirected edge is wired from both
# endpoints, so observed mean degree ≈ 2b.  Paths get shorter as the
# graph gets denser (more edges = more shortcuts).
# ==========================================================================
elif RUN_MODE == 'batch':
    batch_suite(
        complexity_settings=[
            {'b': 2, 'n': 30, 'label': 'sparse  (b=2, b_obs≈4)',
             'param_value': 2, 'param_name': 'Poisson b (b_obs≈2b)'},
            {'b': 4, 'n': 30, 'label': 'medium  (b=4, b_obs≈8)',
             'param_value': 4, 'param_name': 'Poisson b (b_obs≈2b)'},
            {'b': 7, 'n': 30, 'label': 'dense   (b=7, b_obs≈14)',
             'param_value': 7, 'param_name': 'Poisson b (b_obs≈2b)'},
        ],
        agent_factories=RAND_AGENT_FACTORIES,
        n_seeds=5,
        graph_type='random',
    )