import logging
from pathlib import Path

import networkx as nx

from code_gnn.models.embedding_ids import node_type_map, edge_type_map

logger = logging.getLogger(__name__)


def read_csv(filename):
    """
    This is probably faster than pd.read_csv
    though not tested
    """
    with open(filename) as f:
        lines = f.read().splitlines()
        headers = lines.pop(0).split('\t')
        rows = []
        for line in lines:
            row_data = {}
            for i, field in enumerate(line.split('\t')):
                header = headers[i]
                row_data[header] = field
            rows.append(row_data)
        return rows


def read_graph(output_path):
    nodes_path = output_path / 'nodes.csv'
    edges_path = output_path / 'edges.csv'
    assert nodes_path.exists(), f'node file not found: {nodes_path}'
    assert edges_path.exists(), f'edge file not found: {edges_path}'
    nodes_data = read_csv(nodes_path)
    edges_data = read_csv(edges_path)
    return nodes_data, edges_data


def get_cpg(parse_dir, filepath):
    """
    Parse the CPG for a given file that's already parsed using parse_all
    @param parse_dir: the root of the output of parse_all
    @param filepath: the filepath to parse, relative to parse_dir
    
    Return the CPG as networkx graph or None if file is missing or empty.
    """
    parse_dir = Path(parse_dir)
    filepath = Path(filepath)
    assert not filepath.is_absolute(), f'{filepath=} should be a relative path'
    output_path = parse_dir / str(filepath)
    if not output_path.exists():
        return None
    nodes_data, edges_data = read_graph(output_path)
    cpg = nx.MultiDiGraph()
    assert nodes_data[0]["type"] == 'File', 'type of first node must be "File"'
    for na in nodes_data:
        if len(na["code"]) > 0 and na["code"][0] == '"' and na["code"][-1] == '"':
            na["code"] = na["code"][1:-1]
        na["code"] = na["code"].replace('""', '"')
        na.update(
            {"label": f'{na["key"]} ({na["type"]}): {na["code"]}'})  # Graphviz label
        # Cover fault in Joern exposed by tests/acceptance/loop_exchange/chrome_debian/18159_0.c
        if na["type"].endswith('Statement'):
            line, col, offset, end_offset = (int(x) for x in na["location"].split(':'))
            if na["type"] == 'CompoundStatement':
                na["location"] = ':'.join(str(o) for o in (line, col, offset, end_offset))
    nodes = list(zip([int(x["key"]) for x in nodes_data], nodes_data))
    if len(nodes) == 0:
        return None
    cpg.add_nodes_from(nodes)

    # Multigraph
    unique_edge_types = sorted(set(ea["type"] for ea in edges_data))
    edge_type_idx = {et: i for i, et in enumerate(unique_edge_types)}
    for ea in edges_data:
        ea.update({"label": f'({ea["type"]}): {ea["var"]}', "color": edge_type_idx[ea["type"]],
                   "colorscheme": "pastel28"})  # Graphviz label
    edges = [(int(x["start"]), int(x["end"]), x) for x in edges_data]
    if len(edges) == 0:
        return None
    cpg.add_edges_from(edges)

    # Filter out IS_FILE_OF attribute
    edges_to_remove = [(u, v, k) for u, v, k, t in cpg.edges(keys=True, data='type') if t == 'IS_FILE_OF']
    cpg.remove_edges_from(edges_to_remove)
    cpg.remove_nodes_from(list(nx.isolates(cpg)))

    # Remove unneeded node/edge attributes
    for _, attr in cpg.nodes(data=True):
        for key in list(attr.keys()):
            if key == 'code':
                pass
            elif key == 'location':
                location_split = attr[key].split(':')
                if len(location_split) != 4:
                    attr["line"] = None
                else:
                    line, col, offset, end_offset = location_split
                    attr["line"] = int(line)
            elif key == 'type':
                attr[key] = node_type_map[attr[key]]
            # elif key == 'isCFGNode':
            #     attr[key] = bool(attr[key])
            elif key == 'childNum':
                try:
                    attr[key] = int(attr[key])
                except ValueError:
                    attr[key] = None
            else:
                del attr[key]
    for _, _, attr in cpg.edges(data=True):
        for key in list(attr.keys()):
            if key == 'type':
                attr[key] = edge_type_map[attr[key]]
            else:
                del attr[key]

    return cpg
