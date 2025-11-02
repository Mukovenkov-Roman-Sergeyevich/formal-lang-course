from typing import Set, Tuple

import networkx as nx
import scipy.sparse as sp
from pyformlang.cfg import CFG

from project.cfpq import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    if not graph.nodes:
        return set()

    effective_start_nodes = (
        start_nodes if start_nodes is not None else set(graph.nodes())
    )
    effective_final_nodes = (
        final_nodes if final_nodes is not None else set(graph.nodes())
    )

    wcnf = cfg_to_weak_normal_form(cfg)

    nodes = sorted(list(graph.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    matrices = {}
    for var in wcnf.variables:
        matrices[var] = sp.dok_matrix((num_nodes, num_nodes), dtype=bool)

    for prod in wcnf.productions:
        if len(prod.body) == 1:
            term = prod.body[0]
            if term in wcnf.terminals:
                for u, v, data in graph.edges(data=True):
                    if data.get("label") == term.value:
                        u_idx, v_idx = node_to_idx.get(u), node_to_idx.get(v)
                        if u_idx is not None and v_idx is not None:
                            matrices[prod.head][u_idx, v_idx] = True
        elif not prod.body:
            for i in range(num_nodes):
                matrices[prod.head][i, i] = True

    while True:
        changed = False
        for prod in wcnf.productions:
            if len(prod.body) == 2:
                var_a, var_b, var_c = prod.head, prod.body[0], prod.body[1]

                old_nnz = matrices[var_a].nnz

                matrices[var_a] += matrices[var_b] @ matrices[var_c]

                if matrices[var_a].nnz > old_nnz:
                    changed = True

        if not changed:
            break

    result = set()
    start_symbol = wcnf.start_symbol
    if start_symbol not in matrices:
        return result

    final_matrix = matrices[start_symbol]

    rows, cols = final_matrix.nonzero()

    for i, j in zip(rows, cols):
        start_node = nodes[i]
        final_node = nodes[j]

        if start_node in effective_start_nodes and final_node in effective_final_nodes:
            result.add((start_node, final_node))

    return result
