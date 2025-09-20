import pytest
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State

from project.task2 import regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata, tensor_based_rpq


class TestAdjacencyMatrixFA:
    def test_init_with_none(self):
        fa = AdjacencyMatrixFA(automaton=None)
        assert fa.num_states == 0
        assert fa.start_states == set()
        assert fa.final_states == set()
        assert fa.adj_matrices == {}

    def test_init_with_empty_nfa(self):
        empty_nfa = NondeterministicFiniteAutomaton()
        fa = AdjacencyMatrixFA(empty_nfa)
        assert fa.num_states == 0
        assert fa.start_states == set()
        assert fa.final_states == set()
        assert fa.adj_matrices == {}

    def test_init_with_nfa_no_transitions(self):
        nfa = NondeterministicFiniteAutomaton()
        s0, s1 = State(0), State(1)
        nfa.add_start_state(s0)
        nfa.add_final_state(s1)

        fa = AdjacencyMatrixFA(nfa)
        assert fa.num_states == 2
        assert len(fa.start_states) == 1
        assert len(fa.final_states) == 1
        assert fa.adj_matrices == {}

    def test_does_not_accept_on_empty_fa(self):
        fa = AdjacencyMatrixFA()
        assert not fa.accepts([])
        assert not fa.accepts(["a"])

    def test_accepts_empty_word(self):
        dfa1 = regex_to_dfa("a*")
        fa1 = AdjacencyMatrixFA(dfa1)
        assert fa1.accepts([])

        dfa2 = regex_to_dfa("a")
        fa2 = AdjacencyMatrixFA(dfa2)
        assert not fa2.accepts([])

    def test_accepts_unknown_symbol(self):
        fa = AdjacencyMatrixFA(regex_to_dfa("a.b"))
        assert not fa.accepts(["a", "c"])
        assert not fa.accepts(["c"])

    def test_is_empty_for_various_cases(self):
        assert AdjacencyMatrixFA().is_empty()

        nfa_no_start = NondeterministicFiniteAutomaton()
        nfa_no_start.add_final_state(State(0))
        assert AdjacencyMatrixFA(nfa_no_start).is_empty()

        nfa_no_final = NondeterministicFiniteAutomaton()
        nfa_no_final.add_start_state(State(0))
        assert AdjacencyMatrixFA(nfa_no_final).is_empty()

        nfa_no_path = NondeterministicFiniteAutomaton()
        s0, s1 = State(0), State(1)
        nfa_no_path.add_start_state(s0)
        nfa_no_path.add_final_state(s1)
        nfa_no_path.add_transition(s0, "a", s0)
        assert AdjacencyMatrixFA(nfa_no_path).is_empty()

        nfa_accepts_eps = NondeterministicFiniteAutomaton()
        s0 = State(0)
        nfa_accepts_eps.add_start_state(s0)
        nfa_accepts_eps.add_final_state(s0)
        assert not AdjacencyMatrixFA(nfa_accepts_eps).is_empty()


class TestIntersectAutomata:
    def test_intersect_with_empty_automaton(self):
        fa1 = AdjacencyMatrixFA(regex_to_dfa("a|b"))
        empty_fa = AdjacencyMatrixFA()

        intersect1 = intersect_automata(fa1, empty_fa)
        assert intersect1.is_empty()
        assert intersect1.num_states == 0

        intersect2 = intersect_automata(empty_fa, fa1)
        assert intersect2.is_empty()
        assert intersect2.num_states == 0

    def test_intersect_disjoint_alphabets(self):
        fa1 = AdjacencyMatrixFA(regex_to_dfa("a"))
        fa2 = AdjacencyMatrixFA(regex_to_dfa("b"))

        intersect_fa = intersect_automata(fa1, fa2)
        assert not intersect_fa.num_states == 0
        assert not len(intersect_fa.start_states) == 0
        assert not len(intersect_fa.final_states) == 0
        assert intersect_fa.adj_matrices == {}
        assert intersect_fa.is_empty()

    def test_intersect_where_one_has_no_final_states(self):
        fa1 = AdjacencyMatrixFA(regex_to_dfa("a"))

        nfa_no_final = NondeterministicFiniteAutomaton()
        nfa_no_final.add_start_state(State(0))
        nfa_no_final.add_transition(State(0), "a", State(0))
        fa_no_final = AdjacencyMatrixFA(nfa_no_final)

        intersect_fa = intersect_automata(fa1, fa_no_final)
        assert intersect_fa.final_states == set()
        assert intersect_fa.is_empty()


class TestTensorBasedRPQ:
    @pytest.fixture
    def sample_graph(self) -> MultiDiGraph:
        graph = MultiDiGraph()
        graph.add_edges_from(
            [
                (0, 1, {"label": "a"}),
                (1, 2, {"label": "b"}),
                (2, 0, {"label": "c"}),
            ]
        )
        return graph

    def test_empty_graph(self):
        result = tensor_based_rpq("a", MultiDiGraph(), {0}, {1})
        assert result == set()

    def test_empty_start_nodes(self, sample_graph):
        result = tensor_based_rpq("a", sample_graph, set(), {1})
        assert result == set()

    def test_empty_final_nodes(self, sample_graph):
        result = tensor_based_rpq("a", sample_graph, {0}, set())
        assert result == set()

    def test_nodes_not_in_graph(self, sample_graph):
        result = tensor_based_rpq("a", sample_graph, {0, 99}, {1, 100})
        assert result == {(0, 1)}

    def test_regex_alphabet_not_in_graph(self, sample_graph):
        result = tensor_based_rpq("d*", sample_graph, {0, 1, 2}, {0, 1, 2})
        assert result == {(0, 0), (1, 1), (2, 2)}

        result_d = tensor_based_rpq("d", sample_graph, {0, 1, 2}, {0, 1, 2})
        assert result_d == set()

    def test_epsilon_regex(self, sample_graph):
        result = tensor_based_rpq("", sample_graph, {0, 1}, {1, 2})
        assert result == {(1, 1)}

        result2 = tensor_based_rpq("epsilon", sample_graph, {0, 1, 2}, {0, 1, 2})
        assert result2 == {(0, 0), (1, 1), (2, 2)}

    def test_no_path_exists(self, sample_graph):
        result = tensor_based_rpq("a.c", sample_graph, {0}, {0, 1, 2})
        assert result == set()

    def test_simple_path(self, sample_graph):
        result = tensor_based_rpq("a.b", sample_graph, {0}, {2})
        assert result == {(0, 2)}
