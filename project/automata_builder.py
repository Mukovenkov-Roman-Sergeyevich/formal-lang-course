from typing import Set

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    if not regex:
        regex = "epsilon"

    regex_obj = Regex(regex)

    enfa = regex_obj.to_epsilon_nfa()

    dfa = enfa.to_deterministic()

    return dfa.minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    effective_start_states = start_states if start_states else set(graph.nodes())
    effective_final_states = final_states if final_states else set(graph.nodes())

    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label is not None:
            nfa.add_transition(u, label, v)

    for state in effective_start_states:
        nfa.add_start_state(state)

    for state in effective_final_states:
        nfa.add_final_state(state)

    return nfa
