"""
launcher.py
-----------
Graphical launcher for the pathfinding search visualizer.

Run this file instead of main.py to configure a run through a GUI:

    python launcher.py

Panels
------
Mode          : Single run | Batch compare
Graph source  : City graph | Random graph  (single only adds City)
Algorithm(s)  : checkboxes for all 5 agents
Random params : N nodes, b, weight low/high, seed  (shown for random graph)
Batch options : n_seeds, complexity settings (b values for the 3 settings)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_or(var, default):
    try:
        return int(var.get())
    except ValueError:
        return default

def _float_or(var, default):
    try:
        return float(var.get())
    except ValueError:
        return default

def _seed_or_none(var):
    v = var.get().strip()
    if v.lower() in ('', 'none', 'random'):
        return None
    try:
        return int(v)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main launcher window
# ---------------------------------------------------------------------------

class Launcher(tk.Tk):

    AGENTS = ['BFS', 'DFS', 'IDDFS', 'Greedy', 'A*']

    def __init__(self):
        super().__init__()
        self.title('Pathfinding Search — Launcher')
        self.resizable(False, False)

        # ── Top-level mode ──────────────────────────────────────────────────
        self._mode = tk.StringVar(value='single')
        self._graph = tk.StringVar(value='city')

        # ── Agent checkboxes ────────────────────────────────────────────────
        self._agent_vars = {a: tk.BooleanVar(value=(a == 'A*')) for a in self.AGENTS}

        # ── Random graph params ─────────────────────────────────────────────
        self._n          = tk.StringVar(value='25')
        self._b          = tk.StringVar(value='3')
        self._w_low      = tk.StringVar(value='1')
        self._w_high     = tk.StringVar(value='10')
        self._seed       = tk.StringVar(value='42')

        # ── Batch params ────────────────────────────────────────────────────
        self._n_seeds    = tk.StringVar(value='5')
        self._batch_n    = tk.StringVar(value='30')
        self._b1         = tk.StringVar(value='2')
        self._b2         = tk.StringVar(value='4')
        self._b3         = tk.StringVar(value='7')

        self._build_ui()
        self._on_mode_change()   # set initial visibility

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=10, pady=4)
        PAD = dict(padx=10, pady=8)

        # ── Mode selection ──────────────────────────────────────────────
        frm_mode = ttk.LabelFrame(self, text='Mode', padding=8)
        frm_mode.grid(row=0, column=0, columnspan=2, sticky='ew', **PAD)

        ttk.Radiobutton(frm_mode, text='Single run',
                        variable=self._mode, value='single',
                        command=self._on_mode_change).grid(row=0, column=0, padx=12)
        ttk.Radiobutton(frm_mode, text='Batch compare',
                        variable=self._mode, value='batch',
                        command=self._on_mode_change).grid(row=0, column=1, padx=12)

        # ── Graph source ────────────────────────────────────────────────
        self._frm_graph = ttk.LabelFrame(self, text='Graph source', padding=8)
        self._frm_graph.grid(row=1, column=0, columnspan=2, sticky='ew', **PAD)

        self._rb_city = ttk.Radiobutton(
            self._frm_graph, text='City graph  (Kansas — fixed nodes & weights)',
            variable=self._graph, value='city',
            command=self._on_graph_change)
        self._rb_city.grid(row=0, column=0, columnspan=2, sticky='w', padx=6)

        self._rb_rand = ttk.Radiobutton(
            self._frm_graph, text='Random graph',
            variable=self._graph, value='random',
            command=self._on_graph_change)
        self._rb_rand.grid(row=1, column=0, sticky='w', padx=6)

        # ── Random graph parameters ─────────────────────────────────────
        self._frm_rand = ttk.LabelFrame(self, text='Random graph parameters', padding=8)
        self._frm_rand.grid(row=2, column=0, columnspan=2, sticky='ew', **PAD)

        fields = [
            ('Nodes (N)',          self._n,      '10 – 200'),
            ('Branching factor (b)', self._b,    'Poisson mean degree'),
            ('Weight min',         self._w_low,  'integer ≥ 1'),
            ('Weight max',         self._w_high, 'integer ≥ weight min'),
            ('Seed',               self._seed,   'integer, or leave blank for random'),
        ]
        for i, (lbl, var, hint) in enumerate(fields):
            ttk.Label(self._frm_rand, text=lbl).grid(row=i, column=0, sticky='w', **pad)
            ttk.Entry(self._frm_rand, textvariable=var, width=10).grid(row=i, column=1, **pad)
            ttk.Label(self._frm_rand, text=hint,
                      foreground='gray').grid(row=i, column=2, sticky='w', padx=4)

        # ── Algorithm selection ─────────────────────────────────────────
        self._frm_agents = ttk.LabelFrame(self, text='Algorithm(s)', padding=8)
        self._frm_agents.grid(row=3, column=0, columnspan=2, sticky='ew', **PAD)

        self._select_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self._frm_agents, text='Select all',
                        variable=self._select_all_var,
                        command=self._on_select_all).grid(
            row=0, column=0, columnspan=5, sticky='w', padx=6, pady=(0,4))

        ttk.Separator(self._frm_agents, orient='horizontal').grid(
            row=1, column=0, columnspan=5, sticky='ew', pady=2)

        for i, a in enumerate(self.AGENTS):
            ttk.Checkbutton(self._frm_agents, text=a,
                            variable=self._agent_vars[a]).grid(
                row=2, column=i, padx=14, pady=4)

        self._lbl_single_note = ttk.Label(
            self._frm_agents,
            text='Note: single run uses the first checked algorithm.',
            foreground='gray', font=('', 9, 'italic'))
        self._lbl_single_note.grid(row=3, column=0, columnspan=5,
                                   sticky='w', padx=6, pady=(2,0))

        # ── Batch options ───────────────────────────────────────────────
        self._frm_batch = ttk.LabelFrame(self, text='Batch options', padding=8)
        self._frm_batch.grid(row=4, column=0, columnspan=2, sticky='ew', **PAD)

        ttk.Label(self._frm_batch, text='Runs per setting (seeds)').grid(
            row=0, column=0, sticky='w', **pad)
        ttk.Entry(self._frm_batch, textvariable=self._n_seeds, width=6).grid(
            row=0, column=1, **pad)
        ttk.Label(self._frm_batch, text='≥ 5 recommended',
                  foreground='gray').grid(row=0, column=2, sticky='w')

        ttk.Label(self._frm_batch, text='Nodes (N) per setting').grid(
            row=1, column=0, sticky='w', **pad)
        ttk.Entry(self._frm_batch, textvariable=self._batch_n, width=6).grid(
            row=1, column=1, **pad)

        ttk.Separator(self._frm_batch, orient='horizontal').grid(
            row=2, column=0, columnspan=4, sticky='ew', pady=6)

        ttk.Label(self._frm_batch,
                  text='3 complexity settings — Poisson b values:',
                  font=('', 9, 'bold')).grid(
            row=3, column=0, columnspan=4, sticky='w', padx=6)

        for i, (lbl, var) in enumerate([
            ('Sparse  setting b =',  self._b1),
            ('Medium  setting b =',  self._b2),
            ('Dense   setting b =',  self._b3),
        ]):
            ttk.Label(self._frm_batch, text=lbl).grid(
                row=4+i, column=0, sticky='w', **pad)
            ttk.Entry(self._frm_batch, textvariable=var, width=6).grid(
                row=4+i, column=1, **pad)

        # ── Run button ──────────────────────────────────────────────────
        self._btn = ttk.Button(self, text='▶  Run', command=self._on_run,
                               style='Accent.TButton')
        self._btn.grid(row=5, column=0, columnspan=2, pady=14, ipadx=20, ipady=6)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_mode_change(self):
        mode = self._mode.get()
        if mode == 'single':
            self._rb_city.configure(state='normal')
            self._frm_batch.grid_remove()
            self._lbl_single_note.grid()
            self._on_graph_change()   # show/hide rand params based on graph
        else:
            # Batch is random-only — hide graph source params entirely
            self._graph.set('random')
            self._rb_city.configure(state='disabled')
            self._frm_rand.grid_remove()   # always hidden in batch mode
            self._frm_batch.grid()
            self._lbl_single_note.grid_remove()

    def _on_graph_change(self):
        # Only called in single mode
        if self._graph.get() == 'random':
            self._frm_rand.grid()
        else:
            self._frm_rand.grid_remove()

    def _on_select_all(self):
        val = self._select_all_var.get()
        for var in self._agent_vars.values():
            var.set(val)

    def _on_run(self):
        mode = self._mode.get()
        graph = self._graph.get()

        # Collect selected agents
        selected = [a for a in self.AGENTS if self._agent_vars[a].get()]
        if not selected:
            messagebox.showerror('No algorithm selected',
                                 'Please select at least one algorithm.')
            return

        # Validate random params
        n      = _int_or(self._n, 25)
        b      = _int_or(self._b, 3)
        w_low  = _int_or(self._w_low, 1)
        w_high = _int_or(self._w_high, 10)
        seed   = _seed_or_none(self._seed)

        if w_low < 1:
            messagebox.showerror('Invalid weight', 'Weight min must be ≥ 1.')
            return
        if w_high < w_low:
            messagebox.showerror('Invalid weight', 'Weight max must be ≥ weight min.')
            return
        if n < 2:
            messagebox.showerror('Invalid N', 'Need at least 2 nodes.')
            return
        if b < 1:
            messagebox.showerror('Invalid b', 'Branching factor must be ≥ 1.')
            return

        # Batch params
        n_seeds = max(1, _int_or(self._n_seeds, 5))
        batch_n = _int_or(self._batch_n, 30)
        b1 = _int_or(self._b1, 2)
        b2 = _int_or(self._b2, 4)
        b3 = _int_or(self._b3, 7)

        if mode == 'batch':
            bs = sorted([b1, b2, b3])
            if len(set(bs)) < 3:
                messagebox.showerror('Invalid b values',
                                     'The three b settings must be distinct.')
                return

        # Close the launcher and run
        self.destroy()

        if mode == 'single':
            _run_single(graph, selected[0], n, b, (w_low, w_high), seed)
        else:
            _run_batch(selected, batch_n, [b1, b2, b3], n_seeds)


# ---------------------------------------------------------------------------
# Execution functions (called after GUI closes)
# ---------------------------------------------------------------------------

def _make_agent(name, graph):
    from agent import BFSAgent, DFSAgent, IDDFSAgent, GreedyAgent, AStarAgent
    from heuristics import haversine_graph, euclidean_graph
    from graph_sources import CityGraph

    is_city = isinstance(graph, CityGraph)
    h = haversine_graph if is_city else euclidean_graph

    return {
        'BFS':    lambda g=graph: BFSAgent(g),
        'DFS':    lambda g=graph: DFSAgent(g),
        'IDDFS':  lambda g=graph: IDDFSAgent(g),
        'Greedy': lambda g=graph: GreedyAgent(g, heuristic=h),
        'A*':     lambda g=graph: AStarAgent(g, heuristic=h),
    }[name]()


def _run_single(graph_type, agent_name, n, b, weight_range, seed):
    """Build graph, run agent, visualize."""
    from benchmarking import single_run

    if graph_type == 'city':
        from graph_sources import CityGraph
        graph = CityGraph('coordinates.csv', 'Adjacencies.txt', interactive=True)
    else:
        from graph_sources import RandomGraph
        graph = RandomGraph(n=n, b=b, weight_range=weight_range,
                            seed=seed, interactive=True)

    agent = _make_agent(agent_name, graph)
    single_run(agent, visualize=True)


def _run_batch(selected_agents, batch_n, b_values, n_seeds):
    """Build agent factory list from selection and run batch suite."""
    from benchmarking import batch_suite, RAND_AGENT_FACTORIES

    # Filter full factory list to only selected agents
    all_factories = RAND_AGENT_FACTORIES()
    factories_filtered = [(name, fn) for name, fn in all_factories
                          if name in selected_agents]

    # Sort b values ascending for meaningful sparse→dense progression
    b1, b2, b3 = sorted(b_values)

    batch_suite(
        complexity_settings=[
            {'b': b1, 'n': batch_n,
             'label': f'sparse  (b={b1}, b_obs≈{b1*2})',
             'param_value': b1, 'param_name': 'Poisson b (b_obs≈2b)'},
            {'b': b2, 'n': batch_n,
             'label': f'medium  (b={b2}, b_obs≈{b2*2})',
             'param_value': b2, 'param_name': 'Poisson b (b_obs≈2b)'},
            {'b': b3, 'n': batch_n,
             'label': f'dense   (b={b3}, b_obs≈{b3*2})',
             'param_value': b3, 'param_name': 'Poisson b (b_obs≈2b)'},
        ],
        agent_factories=lambda: factories_filtered,
        n_seeds=n_seeds,
        graph_type='random',
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = Launcher()
    app.mainloop()