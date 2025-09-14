import pytest
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)

from project.task2 import regex_to_dfa, graph_to_nfa


def test_regex_to_dfa_empty_string():
    dfa = regex_to_dfa("")
    assert isinstance(dfa, DeterministicFiniteAutomaton)
    assert dfa.accepts([])
    assert not dfa.accepts(["a"])


def test_regex_to_dfa_epsilon_symbol():
    dfa_epsilon = regex_to_dfa("epsilon")
    dfa_dollar = regex_to_dfa("$")
    assert dfa_epsilon.is_equivalent_to(dfa_dollar)
    assert dfa_epsilon.accepts([])
    assert not dfa_epsilon.accepts(["a"])


def test_regex_to_dfa_simple_union():
    dfa = regex_to_dfa("a|b")
    assert dfa.accepts(["a"])
    assert dfa.accepts(["b"])
    assert not dfa.accepts(["c"])
    assert not dfa.accepts(["ab"])
    assert not dfa.accepts([])


def test_regex_to_dfa_kleene_star():
    dfa = regex_to_dfa("a*")
    assert dfa.accepts([])
    assert dfa.accepts(["a"])
    assert dfa.accepts(["a", "a", "a"])
    assert not dfa.accepts(["b"])
    assert not dfa.accepts(["a", "b"])


def test_regex_to_dfa_concatenation():
    dfa = regex_to_dfa("a.b.c")
    assert dfa.accepts(["a", "b", "c"])
    assert not dfa.accepts(["a", "b"])
    assert not dfa.accepts(["abc"])


def test_regex_to_dfa_complex_expression():
    dfa = regex_to_dfa("(a|b)*.c")
    assert dfa.accepts(["c"])
    assert dfa.accepts(["a", "c"])
    assert dfa.accepts(["b", "c"])
    assert dfa.accepts(["a", "b", "a", "c"])
    assert not dfa.accepts([])
    assert not dfa.accepts(["a", "b"])
    assert not dfa.accepts(["c", "a"])


def test_regex_to_dfa_is_minimal():
    dfa_redundant = regex_to_dfa("a|a")
    dfa_simple = regex_to_dfa("a")
    assert dfa_redundant.is_equivalent_to(dfa_simple)
    assert len(dfa_redundant.states) == 2


def test_regex_to_dfa_precedence():
    dfa1 = regex_to_dfa("a.b|c")
    assert dfa1.accepts(["a", "b"])
    assert dfa1.accepts(["c"])
    assert not dfa1.accepts(["a", "c"])

    dfa2 = regex_to_dfa("a.(b|c)")
    assert dfa2.accepts(["a", "b"])
    assert dfa2.accepts(["a", "c"])
    assert not dfa2.accepts(["c"])


def test_graph_to_nfa_empty_graph():
    graph = MultiDiGraph()
    nfa = graph_to_nfa(graph, set(), set())

    assert isinstance(nfa, NondeterministicFiniteAutomaton)
    assert nfa.is_empty()


@pytest.fixture
def single_node_graph() -> MultiDiGraph:
    graph = MultiDiGraph()
    graph.add_node(0)
    return graph


def test_graph_to_nfa_single_node_no_start_final(single_node_graph: MultiDiGraph):
    nfa = graph_to_nfa(single_node_graph, set(), set())

    assert not nfa.is_empty()
    assert nfa.accepts([])
    assert not nfa.accepts(["a"])


def test_graph_to_nfa_single_node_start_only(single_node_graph: MultiDiGraph):
    nfa = graph_to_nfa(single_node_graph, {0}, set())

    assert not nfa.is_empty()
    assert nfa.accepts([])


def test_graph_to_nfa_single_node_final_only(single_node_graph: MultiDiGraph):
    nfa = graph_to_nfa(single_node_graph, set(), {0})

    assert not nfa.is_empty()
    assert nfa.accepts([])


def test_graph_to_nfa_single_node_explicit_start_and_final(
    single_node_graph: MultiDiGraph,
):
    nfa = graph_to_nfa(single_node_graph, {0}, {0})

    assert not nfa.is_empty()
    assert nfa.accepts([])


def test_graph_to_nfa_no_path_between_start_final_is_empty():
    graph = MultiDiGraph()
    graph.add_node(0)
    graph.add_node(1)

    nfa = graph_to_nfa(graph, {0}, {1})

    assert nfa.is_empty()


def test_graph_to_nfa_simple_edge():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")

    nfa = graph_to_nfa(graph, {0}, {1})
    assert nfa.accepts(["a"])
    assert not nfa.accepts([])
    assert not nfa.accepts(["b"])


def test_graph_to_nfa_multiple_edges():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")
    nfa = graph_to_nfa(graph, {0}, {1})
    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])
    assert not nfa.accepts(["c"])


def test_graph_to_nfa_cycle():
    graph = MultiDiGraph()
    graph.add_edge(0, 0, label="loop")

    nfa = graph_to_nfa(graph, {0}, {0})
    assert nfa.accepts([])
    assert nfa.accepts(["loop"])
    assert nfa.accepts(["loop", "loop"])
    assert not nfa.accepts(["a"])


def test_graph_to_nfa_no_start_final_specified():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.accepts([])

    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])

    assert nfa.accepts(["a", "b"])


def test_graph_to_nfa_ignores_unlabeled_edges():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2)

    nfa = graph_to_nfa(graph, {0}, {2})
    assert not nfa.accepts(["a"])
    assert nfa.is_empty()
