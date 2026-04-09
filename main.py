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
    CITY_AGENT_FACTORIES, RAND_AGENT_FACTORIES,
)

# ==========================================================================
# TOP-LEVEL MODE
# ==========================================================================
RUN_MODE   = 'batch'   # 'single' | 'batch'
GRAPH      = 'city'     # for single: 'city' | 'random' | 'grid'
BATCH_GRAPH = 'random'  # for batch:  'city' | 'random'

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
# BATCH COMPARE
# ==========================================================================
elif RUN_MODE == 'batch':

    if BATCH_GRAPH == 'random':
        # 3 complexity settings varying branching factor b
        batch_suite(
            complexity_settings=[
                {'b': 2, 'n': 30, 'label': 'sparse  b=2',
                 'param_value': 2, 'param_name': 'branching factor b'},
                {'b': 4, 'n': 30, 'label': 'medium  b=4',
                 'param_value': 4, 'param_name': 'branching factor b'},
                {'b': 7, 'n': 30, 'label': 'dense   b=7',
                 'param_value': 7, 'param_name': 'branching factor b'},
            ],
            agent_factories=RAND_AGENT_FACTORIES,
            n_seeds=5,
            graph_type='random',
        )

    elif BATCH_GRAPH == 'city':
        # 3 complexity settings varying goal depth (start/goal pairs)
        # Pairs chosen to give short / medium / long solution paths
        batch_suite(
            complexity_settings=[
                {'start': 0,  'goal': 2,  'label': 'shallow  (Abilene→Anthony)',
                 'param_value': 2, 'param_name': 'goal depth (hops)'},
                {'start': 0,  'goal': 30, 'label': 'medium   (Abilene→Newton)',
                 'param_value': 5, 'param_name': 'goal depth (hops)'},
                {'start': 39, 'goal': 10, 'label': 'deep     (Topeka→Coldwater)',
                 'param_value': 9, 'param_name': 'goal depth (hops)'},
            ],
            agent_factories=CITY_AGENT_FACTORIES,
            n_seeds=5,
            graph_type='city',
            coord_path='coordinates.csv',
            adj_path='Adjacencies.txt',
        )