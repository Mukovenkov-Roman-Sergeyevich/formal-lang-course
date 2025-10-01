from typing import Set, Tuple
import scipy.sparse as sparse
from networkx import MultiDiGraph
from project.tensor_rpq import AdjacencyMatrixFA
from project.automata_builder import graph_to_nfa, regex_to_dfa


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: Set[int],
    final_nodes: Set[int],
) -> Set[Tuple[int, int]]:
    if not start_nodes or not final_nodes:
        return set()

    graph_fa = AdjacencyMatrixFA(
        graph_to_nfa(graph, set(graph.nodes()), set(graph.nodes()))
    )
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    if graph_fa.num_states == 0 or regex_fa.num_states == 0:
        return set()

    n_g = graph_fa.num_states
    n_r = regex_fa.num_states
    n_intersect = n_g * n_r

    intersection_adj = sparse.csr_matrix((n_intersect, n_intersect), dtype=bool)
    common_symbols = graph_fa.adj_matrices.keys() & regex_fa.adj_matrices.keys()
    for symbol in common_symbols:
        intersection_adj += sparse.kron(
            graph_fa.adj_matrices[symbol],
            regex_fa.adj_matrices[symbol],
            format="csr",
        )

    frontier = sparse.dok_matrix((n_g, n_intersect), dtype=bool)
    for start_node in start_nodes:
        if start_node not in graph_fa.state_to_idx:
            continue
        start_node_idx = graph_fa.state_to_idx[start_node]
        for start_regex_idx in regex_fa.start_states:
            intersection_idx = start_node_idx * n_r + start_regex_idx
            frontier[start_node_idx, intersection_idx] = True

    frontier = frontier.tocsr()
    reachable = frontier.copy()

    while frontier.nnz > 0:
        next_frontier = frontier @ intersection_adj

        newly_reached = next_frontier - next_frontier.multiply(reachable)

        reachable += newly_reached
        frontier = newly_reached

    result = set()
    rows, cols = reachable.nonzero()

    for i in range(len(rows)):
        start_node_idx = rows[i]
        intersection_idx = cols[i]

        end_node_idx = intersection_idx // n_r
        end_regex_idx = intersection_idx % n_r

        end_node = graph_fa._idx_to_state[end_node_idx]
        if end_node in final_nodes and end_regex_idx in regex_fa.final_states:
            start_node = graph_fa._idx_to_state[start_node_idx]
            result.add((start_node, end_node))

    return result
