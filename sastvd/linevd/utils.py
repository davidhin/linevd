import pandas as pd
import sastvd as svd
import sastvd.helpers.joern as svdj


def ne_groupnodes(n, e):
    """Group nodes with same line number."""
    nl = n.copy()
    el = e.copy()
    nl = nl.dropna(subset=["lineNumber"])
    nl = nl.groupby("lineNumber").head(1)
    nl["nodeId"] = nl.id
    nl.id = nl.lineNumber
    el.innode = el.line_in
    el.outnode = el.line_out
    nl = svdj.drop_lone_nodes(nl, el)
    el = el.drop_duplicates(subset=["innode", "outnode", "etype"])
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    return nl, el


def feature_extraction(_id, graph_type="cfgcdg", return_nodes=False, return_node_ids=False, return_iddict=False, group=True, return_node_types=False):
    """Extract graph feature (basic).

    _id = svddc.svdds.itempath(177775)
    _id = svddc.svdds.itempath(180189)
    _id = svddc.svdds.itempath(178958)

    return_nodes arg is used to get the node information (for empirical evalu
    ation).
    """
    # Get CPG
    n, e = svdj.get_node_edges(_id)
    if group:
        n, e = svd.ne_groupnodes(n, e)
    else:
        n = n[n.lineNumber != ""].copy()
        n.lineNumber = n.lineNumber.astype(int)
        n["nodeId"] = n.id
        e.innode = e.innode.astype(int)
        e.outnode = e.outnode.astype(int)
        n = svdj.drop_lone_nodes(n, e)
        e = e.drop_duplicates(subset=["innode", "outnode", "etype"])

    # Return node metadata
    if return_nodes:
        return n

    # Filter nodes
    e = svdj.rdg(e, graph_type.split("+")[0])
    n = svdj.drop_lone_nodes(n, e)

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index()
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # Map edge types
    etypes = e.etype.tolist()
    d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    etypes = [d[i] for i in etypes]

    # Append function name to code
    if "+raw" not in graph_type:
        pass
    else:
        n.code = "</s>" + " " + n.code

    ret = [n.code.tolist(), n.lineNumber.tolist(), e.innode.tolist(), e.outnode.tolist(), etypes]
    if return_node_ids:
        ret.append(n.nodeId.tolist())
    if return_node_types:
        label = n._label.tolist()
        # TODO: replace calls to <operator>.assignment and such with name attribute
        ret.append(label)
    if return_iddict:
        ret.append(iddict)
    # Return plain-text code, line number list, innodes, outnodes
    return tuple(ret)