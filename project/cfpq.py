from collections import defaultdict, deque
from typing import Set, Tuple

import networkx as nx
from pyformlang.cfg import CFG, Production, Epsilon, Variable, Terminal


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    cnf = cfg.to_normal_form()

    new_productions = cnf.productions.copy()

    nullable_symbols = cfg.get_nullable_symbols()
    for symbol in nullable_symbols:
        new_productions.add(Production(symbol, [Epsilon()]))

    return CFG(
        variables=cnf.variables,
        terminals=cnf.terminals,
        start_symbol=cnf.start_symbol,
        productions=new_productions,
    )


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    wcnf = cfg_to_weak_normal_form(cfg)

    eps_prods = {prod.head for prod in wcnf.productions if not prod.body}
    term_prods = defaultdict(set)
    var_prods = defaultdict(set)

    for prod in wcnf.productions:
        if len(prod.body) == 1 and isinstance(prod.body[0], Terminal):
            term_prods[prod.body[0].value].add(prod.head)
        elif len(prod.body) == 2:
            var1, var2 = prod.body
            if isinstance(var1, Variable) and isinstance(var2, Variable):
                var_prods[(var1, var2)].add(prod.head)

    r = set()
    m = deque()

    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label in term_prods:
            for var in term_prods[label]:
                if (var, u, v) not in r:
                    r.add((var, u, v))
                    m.append((var, u, v))

    for node in graph.nodes():
        for var in eps_prods:
            if (var, node, node) not in r:
                r.add((var, node, node))
                m.append((var, node, node))

    while m:
        var_n, u, v = m.popleft()

        for var_m, w, u_prime in r.copy():
            if u_prime == u:
                prod_body = (var_m, var_n)
                if prod_body in var_prods:
                    for var_p in var_prods[prod_body]:
                        new_triple = (var_p, w, v)
                        if new_triple not in r:
                            r.add(new_triple)
                            m.append(new_triple)

        for var_m, v_prime, w in r.copy():
            if v_prime == v:
                prod_body = (var_n, var_m)
                if prod_body in var_prods:
                    for var_p in var_prods[prod_body]:
                        new_triple = (var_p, u, w)
                        if new_triple not in r:
                            r.add(new_triple)
                            m.append(new_triple)

    result = set()
    start_var = wcnf.start_symbol
    effective_start_nodes = (
        start_nodes if start_nodes is not None else set(graph.nodes())
    )
    effective_final_nodes = (
        final_nodes if final_nodes is not None else set(graph.nodes())
    )

    for var, start_node, final_node in r:
        if (
            var == start_var
            and start_node in effective_start_nodes
            and final_node in effective_final_nodes
        ):
            result.add((start_node, final_node))

    return result
