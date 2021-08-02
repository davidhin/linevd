"""Implementation of IVDetect."""

import networkx as nx
import pandas as pd
import sastvd.helpers.joern as svdj


def feature_extraction(filepath):
    """Extract relevant components of IVDetect Code Representation.

    DEBUGGING:
    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/180189.c"

    PRINTING:
    svdj.plot_graph_node_edge_df(nodes, svdj.rdg(edges, "ast"), [19], 0)
    pd.options.display.max_colwidth = 500
    print(subseq.to_markdown(mode="github"))
    print(nametypes.to_markdown(mode="github"))
    print(undir_edgesline.to_markdown(mode="github"))

    4/5 COMPARISON:
    Theirs: 31, 22, 13, 10, 6, 29, 25, 23
    Ours  : 40, 30, 19, 14, 7, 38, 33, 31
    Pred  : 40,   , 19, 14, 7, 38, 33, 31
    """
    nodes, edges = svdj.get_node_edges(filepath)

    # 1. Generate tokenised subtoken sequences
    subseq = (
        nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    subseq = subseq[["lineNumber", "code", "local_type"]].copy()
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

    # 4/5. Data dependency / Control dependency context
    # Group nodes into statements
    nodesline = (
        nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    edgesline = edges.copy()
    edgesline.innode = edgesline.line_in
    edgesline.outnode = edgesline.line_out
    nodesline.id = nodesline.lineNumber
    edgesline = svdj.rdg(edgesline, "pdg")
    nodesline = svdj.drop_lone_nodes(nodesline, edgesline)
    # Drop duplicate edges
    edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])
    # REACHING DEF to DDG
    edgesline["etype"] = edgesline.apply(
        lambda x: f"DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
    )
    edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
    edgesline_reverse.columns = ["outnode", "innode", "etype"]
    undir_edgesline = pd.concat([edgesline, edgesline_reverse])
    undir_edgesline = undir_edgesline[undir_edgesline.innode != undir_edgesline.outnode]
    undir_edgesline = undir_edgesline.groupby(["innode", "etype"]).agg({"outnode": set})
    undir_edgesline = undir_edgesline.reset_index()
    undir_edgesline = undir_edgesline.pivot("innode", "etype", "outnode")
    undir_edgesline = undir_edgesline.reset_index()[["innode", "CDG", "DDG"]]
    undir_edgesline.columns = ["lineNumber", "control", "data"]

    return subseq, ast, nametypes, undir_edgesline
