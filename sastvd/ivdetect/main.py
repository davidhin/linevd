"""Implementation of IVDetect."""

import networkx as nx
import pandas as pd
import sastvd.helpers.joern as svdj


def feature_extraction(filepath, lineNumber: list = [], hop: int = 1):
    """Extract relevant components of IVDetect Code Representation."""
    nodes, edges = svdj.get_node_edges(filepath)

    # 1. Generate tokenised subtoken sequences
    subseq = (
        nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    subseq = subseq[["lineNumber", "code", "local_type"]].copy()
    subseq.code = subseq.apply(
        lambda x: x.code + ";" if len(x.local_type) > 0 else x.code, axis=1
    )
    subseq.code = subseq.local_type + " " + subseq.code
    subseq = subseq.drop(columns="local_type")
    subseq = subseq[~subseq.eq("").any(1)]
    subseq = subseq[subseq.code != " "]
    subseq.lineNumber = subseq.lineNumber.astype(int)
    subseq = subseq.sort_values("lineNumber")

    # 2. Line to AST
    ast_edges = svdj.rdg(edges, "ast")[["innode", "outnode", "etype"]]
    ast_nodes = svdj.drop_lone_nodes(nodes, ast_edges)[["id", "_label", "code"]]
    ast = {"nodes": ast_nodes, "edges": ast_edges}

    # 3. Variable names and types
    reftype_edges = svdj.rdg(edges, "reftype")
    reftype_nodes = svdj.drop_lone_nodes(nodes, reftype_edges)
    reftype_nx = nx.Graph()
    reftype_nx.add_edges_from(reftype_edges[["innode", "outnode"]].to_numpy())
    reftype_cc = list(nx.connected_components(reftype_nx))
    varnametypes = list()
    for cc in reftype_cc:
        cc_nodes = reftype_nodes[reftype_nodes.id.isin(cc)]
        var_type = cc_nodes[cc_nodes["_label"] == "TYPE"].name.item()
        for idrow in cc_nodes[cc_nodes["_label"] == "IDENTIFIER"].itertuples():
            varnametypes += [[idrow.lineNumber, var_type, idrow.name]]
    nametypes = pd.DataFrame(varnametypes, columns=["lineNumber", "type", "name"])
    nametypes = nametypes.drop_duplicates().sort_values("lineNumber")

    return subseq, ast, nametypes

    # BLAH BLAH BLAH

    # Group nodes into statements
    # nodes = (
    #     nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    #     .groupby("lineNumber")
    #     .head(1)
    # )
    # edges.innode = edges.line_in
    # edges.outnode = edges.line_out
    # nodes.id = nodes.lineNumber

    # Filter to PDG/AST
    # edges = svdj.rdg(edges, "ast")
    edges = svdj.rdg(edges, "reftype")
    # edges = svdj.rdg(edges, "pdg")

    # Drop duplicate edges
    edges = edges.drop_duplicates(subset=["innode", "outnode", "etype"])
    # REACHING DEF to DDG
    edges["etype"] = edges.apply(
        lambda x: f"DDG: {x.dataflow}" if x.etype == "REACHING_DEF" else x.etype, axis=1
    )
    edges = edges[edges.innode != edges.outnode]

    nodeids = nodes[nodes.lineNumber.isin(lineNumber)].id
    if len(lineNumber) > 0:
        keep_nodes = svdj.neighbour_nodes(nodes, edges, nodeids, hop)
        keep_nodes = set(list(nodeids) + [i for j in keep_nodes.values() for i in j])
        nodes = nodes[nodes.id.isin(keep_nodes)]
        edges = edges[
            (edges.innode.isin(keep_nodes)) & (edges.outnode.isin(keep_nodes))
        ]

    # Visualise graph
    svdj.plot_graph_node_edge_df(nodes, edges)


filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/180189.c"
# filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/177860.c"
# lineNumber = [36]
# hop = 1
# feature_extraction(filepath, lineNumber, hop)
feature_extraction(filepath)


# # Get line number
# nodes_new = nodes_new.reset_index().rename(columns={"index": "adj"})
# id2adj = pd.Series(nodes_new.id.values, index=nodes_new.adj).to_dict()
# arr = [
#     list(i)
#     for i in zip(edges.innode.map(id2adj), edges.outnode.map(id2adj))
# ]
# arr = np.array(arr)
# shape = tuple(arr.max(axis=0)[:2] + 1)
# coo = sparse.coo_matrix((np.ones(len(arr)), (arr[:, 0], arr[:, 1])), shape=shape)
# csr = coo.tocsr()
