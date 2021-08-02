"""Implementation of IVDetect."""

import sastvd.helpers.joern as svdj


def feature_extraction(filepath, lineNumber: list = [], hop: int = 1):
    """Extract relevant components of IVDetect Code Representation."""
    nodes, edges = svdj.get_node_edges(filepath)

    # 1. Generate tokenised subtoken sequences
    f1_subseq = (
        nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    f1_subseq = f1_subseq[["lineNumber", "code", "local_type"]].copy()
    f1_subseq.code = f1_subseq.local_type + " " + f1_subseq.code + ";"
    f1_subseq = f1_subseq.drop(columns="local_type")
    f1_subseq = f1_subseq[~f1_subseq.eq("").any(1)]
    f1_subseq = f1_subseq[f1_subseq.code != " "]
    f1_subseq.lineNumber = f1_subseq.lineNumber.astype(int)
    f1_subseq = f1_subseq.sort_values("lineNumber")

    # 2. Line to AST
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

    # Drop lone nodes
    nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]

    # Visualise graph
    dot = svdj.get_digraph(
        nodes[["id", "node_label"]].to_numpy().tolist(),
        edges[["outnode", "innode", "etype"]].to_numpy().tolist(),
    )
    dot.render("/tmp/tmp.gv", view=True)


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
