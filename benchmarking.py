"""
benchmarking.py
---------------
Batch benchmarking harness for search agents.

Modes
-----
single_run(agent, visualize=True)
    Run one agent on its already-configured graph, display the animation
    with metrics panel, and return the MetricsResult.  Thin wrapper kept
    here so main.py has a single import for both modes.

batch_compare(graph_factory, agent_factories, n_seeds, seeds, label)
    Run every agent × every seed, collect MetricsResult for each, and
    return a BatchReport.

batch_suite(complexity_settings, agent_factories, n_seeds)
    Run batch_compare across 3 complexity settings (varying b or graph
    size), collect all BatchReports, and produce the final comparison
    table + charts.

Classes
-------
BatchReport
    Holds all MetricsResult objects for one (graph_config × agents × seeds)
    combination.  Computes mean ± std across seeds.

    .table()      → prints a formatted comparison table to stdout
    .chart(...)   → draws bar charts (runtime, memory) and a line chart
                    (nodes expanded vs complexity parameter)

Usage (see main.py for full examples)
--------------------------------------
    from benchmarking import batch_suite, CITY_AGENT_FACTORIES, RAND_AGENT_FACTORIES

    # 3 complexity settings on random graphs, all agents, 5 seeds each
    batch_suite(
        complexity_settings=[
            {'b': 2, 'n': 30, 'label': 'sparse  b=2'},
            {'b': 4, 'n': 30, 'label': 'medium  b=4'},
            {'b': 7, 'n': 30, 'label': 'dense   b=7'},
        ],
        agent_factories=RAND_AGENT_FACTORIES,
        n_seeds=5,
    )
"""

import statistics
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np

from metrics import run_with_metrics


# ---------------------------------------------------------------------------
# Pre-built agent factory lists
# ---------------------------------------------------------------------------

def _city_factories():
    """Return agent factory list for CityGraph."""
    from agent import BFSAgent, DFSAgent, IDDFSAgent, GreedyAgent, AStarAgent
    from heuristics import haversine_graph
    return [
        ('BFS',    lambda g: BFSAgent(g)),
        ('DFS',    lambda g: DFSAgent(g)),
        ('IDDFS',  lambda g: IDDFSAgent(g)),
        ('Greedy', lambda g: GreedyAgent(g, heuristic=haversine_graph)),
        ('A*',     lambda g: AStarAgent(g,  heuristic=haversine_graph)),
    ]

def _rand_factories():
    """Return agent factory list for RandomGraph."""
    from agent import BFSAgent, DFSAgent, IDDFSAgent, GreedyAgent, AStarAgent
    from heuristics import euclidean_graph
    return [
        ('BFS',    lambda g: BFSAgent(g)),
        ('DFS',    lambda g: DFSAgent(g)),
        ('IDDFS',  lambda g: IDDFSAgent(g)),
        ('Greedy', lambda g: GreedyAgent(g, heuristic=euclidean_graph)),
        ('A*',     lambda g: AStarAgent(g,  heuristic=euclidean_graph)),
    ]

# Callable so factories are built fresh each call (avoids closure issues)
# Note: CITY_AGENT_FACTORIES removed — city graphs are deterministic, batch
# benchmarking only applies to RandomGraph.
RAND_AGENT_FACTORIES = _rand_factories


# ---------------------------------------------------------------------------
# BatchReport
# ---------------------------------------------------------------------------

class BatchReport:
    """
    Aggregated results for one complexity setting × all agents × N seeds.

    Parameters
    ----------
    label       : str   human-readable description of this setting
    param_value : float the complexity parameter value (b, density, etc.)
    param_name  : str   name of the complexity parameter ('b', 'density', …)
    results     : dict  agent_name → list[MetricsResult]
    """

    def __init__(self, label, param_value, param_name, results: dict):
        self.label       = label
        self.param_value = param_value
        self.param_name  = param_name
        self.results     = results   # {agent_name: [MetricsResult, ...]}
        self.stats       = self._compute_stats()

    def _compute_stats(self) -> dict:
        """
        Compute mean ± std for each metric across seeds.

        Returns
        -------
        dict  agent_name → {metric_name: (mean, std)}
        """
        metrics_of_interest = [
            'runtime_ms', 'process_kb', 'nodes_generated',
            'nodes_expanded', 'branch_avg', 'solution_depth', 'path_cost',
        ]
        stats = {}
        for agent_name, runs in self.results.items():
            stats[agent_name] = {}
            dicts = [r.as_dict() for r in runs]
            for m in metrics_of_interest:
                vals = [d[m] for d in dicts if d[m] is not None]
                if vals:
                    mean = statistics.mean(vals)
                    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    stats[agent_name][m] = (mean, std)
                else:
                    stats[agent_name][m] = (float('nan'), float('nan'))
        return stats

    def table(self) -> str:
        """Return a formatted comparison table string."""
        agents = list(self.stats.keys())
        cols   = ['runtime_ms', 'process_kb', 'nodes_expanded', 'branch_avg',
                  'solution_depth', 'path_cost']
        hdrs   = ['Agent', 'Time(ms)', 'Mem(KB)', 'Expanded', 'B-avg',
                  'Depth', 'Cost']

        col_w = [10, 14, 10, 10, 8, 7, 10]
        sep   = '  '.join('─' * w for w in col_w)

        lines = [
            f'\n┌─ {self.label} ─┐',
            '  '.join(f'{h:<{w}}' for h, w in zip(hdrs, col_w)),
            sep,
        ]
        for a in agents:
            s    = self.stats[a]
            row  = [a]
            for c in cols:
                mean, std = s.get(c, (float('nan'), 0))
                if c in ('runtime_ms', 'process_kb', 'path_cost', 'branch_avg'):
                    row.append(f'{mean:.2f}±{std:.2f}')
                else:
                    row.append(f'{mean:.1f}±{std:.1f}')
            lines.append('  '.join(f'{v:<{w}}' for v, w in zip(row, col_w)))
        lines.append(sep)
        return '\n'.join(lines)

    def print_table(self):
        print(self.table())


# ---------------------------------------------------------------------------
# Single-run wrapper
# ---------------------------------------------------------------------------

def single_run(agent, visualize: bool = True):
    """
    Run one agent, collect metrics, optionally animate.

    For IDDFSAgent: metrics are collected via search() (pure benchmarking
    mode, final iteration only).  If visualizing, search_verbose() is then
    called separately so the animation shows each deepening iteration.

    Parameters
    ----------
    agent     : any Agent subclass (already configured with a grid/graph)
    visualize : bool  if True, open the search animation window

    Returns
    -------
    MetricsResult
    """
    from agent import IDDFSAgent
    result = run_with_metrics(agent, method='search')

    if visualize:
        # IDDFS needs search_verbose() so the animator has per-iteration events
        if isinstance(agent, IDDFSAgent):
            agent.search_verbose()
        from search_visualization import Visualizer
        Visualizer(agent).show_all(metrics=result)

    return result


# ---------------------------------------------------------------------------
# Batch compare — one complexity setting
# ---------------------------------------------------------------------------

def batch_compare(graph_factory, agent_factories,
                  n_seeds: int = 5,
                  seeds: list | None = None,
                  label: str = '',
                  param_value: float = 0,
                  param_name: str = 'b') -> BatchReport:
    """
    Run all agents on N seeds of the same graph configuration.

    Parameters
    ----------
    graph_factory   : callable(seed) → graph
        Called once per seed; must return a fully configured graph with
        .start and .goal already set.
    agent_factories : list of (name, callable(graph) → agent)
    n_seeds         : int   number of seeds to run (default 5)
    seeds           : list  explicit seed list; if None, uses range(n_seeds)
    label           : str   description for table/chart headers
    param_value     : float complexity parameter value (for chart x-axis)
    param_name      : str   name of the complexity parameter

    Returns
    -------
    BatchReport
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    results = {name: [] for name, _ in agent_factories}

    total = len(seeds) * len(agent_factories)
    done  = 0
    print(f'\n  Running: {label}')

    for seed in seeds:
        graph = graph_factory(seed)
        for agent_name, make_agent in agent_factories:
            agent = make_agent(graph)
            # Always use search() for benchmarking — search_verbose() accumulates
            # all iterations' events which inflates nodes_expanded and memory.
            result = run_with_metrics(agent, method='search')
            results[agent_name].append(result)
            done += 1
            print(f'    [{done}/{total}] {agent_name} seed={seed} '
                  f'rt={result.runtime_s*1000:.2f}ms '
                  f'exp={result.nodes_expanded} '
                  f'cost={result.path_cost:.1f}')

    return BatchReport(label, param_value, param_name, results)


# ---------------------------------------------------------------------------
# Batch suite — 3 complexity settings
# ---------------------------------------------------------------------------

def batch_suite(complexity_settings: list,
                agent_factories,
                n_seeds: int = 5,
                graph_type: str = 'random') -> list[BatchReport]:
    """
    Run batch_compare across multiple complexity settings and produce
    a comparison table and charts.

    Only random graphs are supported — city graphs are deterministic so
    repeat runs with different seeds produce identical results.

    Parameters
    ----------
    complexity_settings : list of dicts with keys:
        'b'           — Poisson branching factor
        'n'           — number of nodes
        'label'       — display label for table/chart
        'param_value' — x-axis value for line chart
        'param_name'  — x-axis label

    agent_factories : callable returning list of (name, factory) tuples
        Pass RAND_AGENT_FACTORIES.

    n_seeds : int   runs per setting per agent (default 5)

    Returns
    -------
    list[BatchReport]
    """
    reports = []
    factories = agent_factories()  # build fresh

    for setting in complexity_settings:
        label       = setting.get('label', str(setting))
        param_value = setting.get('param_value', 0)
        param_name  = setting.get('param_name', 'b')

        if graph_type == 'random':
            n            = setting.get('n', 25)
            b            = setting.get('b', 3)
            weight_range = setting.get('weight_range', (1, 10))
            connectedness = setting.get('connectedness', 1.0)

            def make_graph(seed, _n=n, _b=b, _wr=weight_range, _c=connectedness):
                from graph_sources import RandomGraph
                import random as _random
                rng = _random.Random(seed)
                g = RandomGraph(n=_n, b=_b, weight_range=_wr,
                                connectedness=_c, seed=seed,
                                interactive=False)
                # Pick start/goal: most-separated passable pair by graph distance
                import networkx as nx
                nodes = list(g.G.nodes())
                # Try up to 20 random pairs, pick the one with longest shortest path
                best_pair, best_len = (nodes[0], nodes[-1]), 0
                for _ in range(20):
                    s, t = rng.sample(nodes, 2)
                    try:
                        d = nx.shortest_path_length(g.G, s, t)
                        if d > best_len:
                            best_len = d
                            best_pair = (s, t)
                    except nx.NetworkXNoPath:
                        pass
                g.start, g.goal = best_pair
                return g

        else:
            raise ValueError(f"Unknown graph_type: {graph_type!r}  (only 'random' supported for batch)")

        report = batch_compare(
            graph_factory   = make_graph,
            agent_factories = factories,
            n_seeds         = n_seeds,
            label           = label,
            param_value     = param_value,
            param_name      = param_name,
        )
        reports.append(report)
        report.print_table()

    # ---- Charts --------------------------------------------------------------
    _draw_charts(reports, factories)

    return reports


# ---------------------------------------------------------------------------
# Chart drawing
# ---------------------------------------------------------------------------

def _draw_charts(reports: list, factories: list):
    """
    Produce two figures:
      Fig 1 — Bar charts: mean runtime and mean process memory per agent
               across the 3 complexity settings (grouped bars).
      Fig 2 — Line chart: mean nodes expanded vs complexity parameter
               per agent.
    """
    agent_names  = [name for name, _ in factories]
    setting_lbls = [r.label for r in reports]
    x_vals       = [r.param_value for r in reports]
    param_name   = reports[0].param_name if reports else 'param'

    n_agents   = len(agent_names)
    n_settings = len(reports)
    palette    = plt.cm.tab10.colors

    # ── Figure 1: grouped bar charts ────────────────────────────────────────
    fig1, (ax_rt, ax_mem) = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Algorithm Comparison — Runtime & Memory', fontsize=14, fontweight='bold')

    bar_w  = 0.7 / n_agents
    x_base = np.arange(n_settings)

    for i, (agent_name, color) in enumerate(zip(agent_names, palette)):
        rt_means, rt_stds   = [], []
        mem_means, mem_stds = [], []
        for report in reports:
            s = report.stats.get(agent_name, {})
            rt_m,  rt_s  = s.get('runtime_ms', (0, 0))
            mem_m, mem_s = s.get('process_kb', (0, 0))
            rt_means.append(rt_m);  rt_stds.append(rt_s)
            mem_means.append(mem_m); mem_stds.append(mem_s)

        offset = (i - n_agents / 2 + 0.5) * bar_w
        ax_rt.bar(x_base + offset, rt_means, bar_w * 0.9,
                  yerr=rt_stds, capsize=3,
                  color=color, alpha=0.85, label=agent_name,
                  error_kw={'linewidth': 1, 'ecolor': '#555'})
        ax_mem.bar(x_base + offset, mem_means, bar_w * 0.9,
                   yerr=mem_stds, capsize=3,
                   color=color, alpha=0.85, label=agent_name,
                   error_kw={'linewidth': 1, 'ecolor': '#555'})

    for ax, ylabel, title in [
        (ax_rt,  'Mean runtime (ms)',    'Runtime  (mean ± std)'),
        (ax_mem, 'Mean process mem (KB)', 'Process memory  (mean ± std)'),
    ]:
        ax.set_xticks(x_base)
        ax.set_xticklabels(setting_lbls, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    # ── Figure 2: line chart — nodes expanded vs complexity parameter ────────
    fig2, ax_exp = plt.subplots(figsize=(9, 5))
    fig2.suptitle(f'Nodes Expanded vs {param_name}', fontsize=14, fontweight='bold')

    for i, (agent_name, color) in enumerate(zip(agent_names, palette)):
        means, stds = [], []
        for report in reports:
            s = report.stats.get(agent_name, {})
            m, s_ = s.get('nodes_expanded', (0, 0))
            means.append(m); stds.append(s_)

        means = np.array(means)
        stds  = np.array(stds)
        ax_exp.plot(x_vals, means, marker='o', color=color,
                    linewidth=2, label=agent_name)
        ax_exp.fill_between(x_vals, means - stds, means + stds,
                            color=color, alpha=0.15)

    ax_exp.set_xlabel(param_name, fontsize=11)
    ax_exp.set_ylabel('Mean nodes expanded', fontsize=11)
    ax_exp.set_title(f'Search effort vs {param_name}  (shaded = ±1 std)',
                     fontsize=11, fontweight='bold')
    ax_exp.legend(fontsize=9)
    ax_exp.grid(alpha=0.3)
    ax_exp.set_axisbelow(True)
    if len(x_vals) > 1:
        ax_exp.set_xticks(x_vals)
        ax_exp.set_xticklabels([str(v) for v in x_vals])

    plt.tight_layout()
    plt.show()