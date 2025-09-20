from typing import Iterable, Set, Tuple
import scipy.sparse as sparse
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol,
)
from scipy.sparse.csgraph import shortest_path

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton = None):
        if automaton is None:
            self.num_states = 0
            self.start_states = set()
            self.final_states = set()
            self.adj_matrices = {}
            self.state_to_idx = {}
            self._idx_to_state = {}
            return

        states = sorted(list(automaton.states), key=hash)
        self.state_to_idx = {state: i for i, state in enumerate(states)}
        self._idx_to_state = {i: state for state, i in self.state_to_idx.items()}
        self.num_states = len(states)

        self.start_states = {self.state_to_idx[s] for s in automaton.start_states}
        self.final_states = {self.state_to_idx[s] for s in automaton.final_states}

        self.adj_matrices = {}
        for start_state, transitions in automaton.to_dict().items():
            for symbol, end_states in transitions.items():
                if symbol not in self.adj_matrices:
                    self.adj_matrices[symbol] = sparse.dok_matrix(
                        (self.num_states, self.num_states), dtype=bool
                    )
                if not isinstance(end_states, set):
                    end_states = {end_states}
                for end_state in end_states:
                    start_idx = self.state_to_idx[start_state]
                    end_idx = self.state_to_idx[end_state]
                    self.adj_matrices[symbol][start_idx, end_idx] = True

        for symbol in self.adj_matrices:
            self.adj_matrices[symbol] = self.adj_matrices[symbol].tocsr()

    def accepts(self, word: Iterable[Symbol]) -> bool:
        word_list = list(word)
        if self.num_states == 0:
            return False

        if not word_list:
            return bool(self.start_states & self.final_states)

        current_states_vec = sparse.lil_matrix((1, self.num_states), dtype=bool)
        for state_idx in self.start_states:
            current_states_vec[0, state_idx] = True

        current_states_vec = current_states_vec.tocsr()

        for symbol in word_list:
            if symbol not in self.adj_matrices:
                return False
            current_states_vec = current_states_vec @ self.adj_matrices[symbol]

        reachable_indices = set(current_states_vec.nonzero()[1])
        return not self.final_states.isdisjoint(reachable_indices)

    def is_empty(self) -> bool:
        if self.num_states == 0 or not self.start_states or not self.final_states:
            return True

        if self.start_states & self.final_states:
            return False

        if not self.adj_matrices:
            return True

        total_adj = sparse.csr_matrix((self.num_states, self.num_states), dtype=bool)
        for matrix in self.adj_matrices.values():
            total_adj += matrix

        if total_adj.nnz == 0:
            return True

        dist_matrix = shortest_path(csgraph=total_adj, directed=True, unweighted=True)

        for start_idx in self.start_states:
            for final_idx in self.final_states:
                if dist_matrix[start_idx, final_idx] != float("inf"):
                    return False
        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    result = AdjacencyMatrixFA()
    n1, n2 = automaton1.num_states, automaton2.num_states
    result.num_states = n1 * n2

    result.start_states = {
        s1 * n2 + s2 for s1 in automaton1.start_states for s2 in automaton2.start_states
    }
    result.final_states = {
        s1 * n2 + s2 for s1 in automaton1.final_states for s2 in automaton2.final_states
    }

    common_symbols = automaton1.adj_matrices.keys() & automaton2.adj_matrices.keys()
    for symbol in common_symbols:
        m1 = automaton1.adj_matrices[symbol]
        m2 = automaton2.adj_matrices[symbol]
        result.adj_matrices[symbol] = sparse.kron(m1, m2, format="csr")

    return result


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: Set[int],
    final_nodes: Set[int],
) -> Set[Tuple[int, int]]:
    if not start_nodes or not final_nodes:
        return set()

    graph_nfa = graph_to_nfa(graph, set(graph.nodes()), set(graph.nodes()))
    graph_fa = AdjacencyMatrixFA(graph_nfa)

    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    if graph_fa.num_states == 0 or regex_fa.num_states == 0:
        return set()

    intersection_fa = intersect_automata(graph_fa, regex_fa)

    if not intersection_fa.start_states or not intersection_fa.final_states:
        return set()

    intersection_total_adj = sparse.csr_matrix(
        (intersection_fa.num_states, intersection_fa.num_states), dtype=bool
    )
    if intersection_fa.adj_matrices:
        for matrix in intersection_fa.adj_matrices.values():
            intersection_total_adj += matrix

    dist_matrix = shortest_path(
        csgraph=intersection_total_adj, directed=True, unweighted=True
    )

    result = set()
    n2 = regex_fa.num_states

    for start_node in start_nodes:
        if start_node not in graph_fa.state_to_idx:
            continue
        for final_node in final_nodes:
            if final_node not in graph_fa.state_to_idx:
                continue

            start_idx_g = graph_fa.state_to_idx[start_node]
            final_idx_g = graph_fa.state_to_idx[final_node]

            for start_idx_r in regex_fa.start_states:
                for final_idx_r in regex_fa.final_states:
                    start_idx_intersect = start_idx_g * n2 + start_idx_r
                    final_idx_intersect = final_idx_g * n2 + final_idx_r

                    if (
                        start_idx_intersect >= intersection_fa.num_states
                        or final_idx_intersect >= intersection_fa.num_states
                    ):
                        continue

                    if dist_matrix[start_idx_intersect, final_idx_intersect] != float(
                        "inf"
                    ):
                        result.add((start_node, final_node))
                        break
                else:
                    continue
                break
    return result
