# utils/dag_utils.py

import numpy as np
import networkx as nx

def generate_er_dag(p, edge_prob=0.1, seed=None):
    """
    Generate an Erdős–Rényi DAG by orienting an undirected ER graph.
    """
    rng = np.random.default_rng(seed)
    undirected = nx.erdos_renyi_graph(p, edge_prob, seed=seed)
    ordering = rng.permutation(p)
    dag = nx.DiGraph()

    dag.add_nodes_from(range(p))
    for u, v in undirected.edges():
        if ordering[u] < ordering[v]:
            dag.add_edge(u, v)
        else:
            dag.add_edge(v, u)
    return dag


def generate_scale_free_dag(p, seed=None):
    """
    Generate a scale-free DAG by orienting BA graph.
    """
    rng = np.random.default_rng(seed)
    undirected = nx.barabasi_albert_graph(p, m=2, seed=seed)
    ordering = rng.permutation(p)
    dag = nx.DiGraph()

    dag.add_nodes_from(range(p))
    for u, v in undirected.edges():
        if ordering[u] < ordering[v]:
            dag.add_edge(u, v)
        else:
            dag.add_edge(v, u)
    return dag


def dag_to_adjacency(dag):
    """Return weighted adjacency matrix B with 1s for edges."""
    p = dag.number_of_nodes()
    B = np.zeros((p, p))
    for u, v in dag.edges():
        B[u, v] = 1
    return B
