import shutil

import networkx as nx

from code_gnn.analysis.cpg import parse_all, get_cpg
from code_gnn.analysis.dataflow import ReachingDefinitions
from code_gnn.globals import test_data_dir


def test_simple():
    my_test_dir = test_data_dir / 'dataflow_test'
    parsed_dir = my_test_dir / 'parsed'
    src_dir = my_test_dir / 'files'
    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)
    parse_all(parsed_dir, src_dir, [src_dir / 'test.c'])
    cpg = get_cpg(parsed_dir, 'test.c')
    problem = ReachingDefinitions(cpg)

    reaching_defs = problem.get_reaching_definitions()

    for n, rd in sorted(reaching_defs.items(), key=lambda x: x[0]):
        print(f'RDin({n}):')
        vars_to_defs = {}
        for v, d in rd:
            if v not in vars_to_defs:
                vars_to_defs[v] = [d]
            else:
                vars_to_defs[v].append(d)
        for v in vars_to_defs:
            print(f'{v}: {[cpg.nodes[d]["code"] for d in vars_to_defs[v]]}')

    print('Inputs:')
    for node in problem.cfg:
        gen = problem.gen(node)
        print(f'"{cpg.nodes[node]["code"]} ({node})": {gen}')
    print('Variables:', problem.variables)
    print(nx.nx_pydot.to_pydot(cpg))


def test_null_pointer():
    my_test_dir = test_data_dir / 'npdr_test'
    parsed_dir = my_test_dir / 'parsed'
    src_dir = my_test_dir / 'files'
    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)
    parse_all(parsed_dir, src_dir, [src_dir / 'test.c'])
    cpg = get_cpg(parsed_dir, 'test.c')
    problem = ReachingDefinitions(cpg)

    reaching_defs = problem.get_reaching_definitions()

    for n, rd in sorted(reaching_defs.items(), key=lambda x: x[0]):
        # if len(rd) == 0:
        #     continue
        print(f'RDin({n}) "{cpg.nodes[n]["code"]}":')
        vars_to_defs = {}
        for v, d in rd:
            if v not in vars_to_defs:
                vars_to_defs[v] = [d]
            else:
                vars_to_defs[v].append(d)
        for v in vars_to_defs:
            print(f'\t{v}: {[cpg.nodes[d]["code"] for d in vars_to_defs[v]]}')
