from importlib import reload

import sastvd.helpers.dclass as svddc
import sastvd.helpers.joern as svdj
import sastvd.linevd as lvd
from graphviz import Digraph

reload(svdj)


def get_digraph(nodes, edges, edge_label=True):
    """Plote digraph given nodes and edges list."""
    dot = Digraph(comment="Combined PDG", engine="neato")

    nodes = [n + [svdj.nodelabel2line(n[1])] for n in nodes]
    colormap = {"": "white"}
    for n in nodes:
        if n[2] not in colormap:
            colormap[n[2]] = svdj.randcolor()

    for n in nodes:
        style = {"shape": "circle", "fixedsize": "true", "width": "0.5"}
        dot.node(str(n[0]), str(n[1]), **style)
    for e in edges:
        style = {"color": "black"}
        if e[2] == "CALL":
            style["style"] = "solid"
            style["color"] = "purple"
        elif e[2] == "AST":
            style["style"] = "solid"
            style["color"] = "black"
        elif e[2] == "CFG":
            style["style"] = "solid"
            style["color"] = "red"
        elif e[2] == "CDG":
            style["style"] = "solid"
            style["color"] = "black"
        elif e[2] == "REACHING_DEF":
            style["style"] = "dashed"
            style["color"] = "black"
        elif "DDG" in e[2]:
            style["style"] = "dashed"
            style["color"] = "red"
            # style["dir"] = "back"
        else:
            style["style"] = "solid"
            style["color"] = "black"
        style["penwidth"] = "1"
        if edge_label:
            dot.edge(str(e[0]), str(e[1]), e[2], **style)
        else:
            dot.edge(str(e[0]), str(e[1]), **style)
    return dot


_id = svddc.BigVulDataset.itempath(182352)

lineMap = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    12: 11,
    13: 12,
    14: 13,
    16: 14,
    17: 15,
    21: 18,
    22: 19,
    25: 21,
    26: 22,
    29: 25,
    31: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
}


# Get CPG
n, e = svdj.get_node_edges(_id)
n.lineNumber = n.lineNumber.map(lineMap).fillna("")
e.line_in = e.line_in.map(lineMap).fillna("")
e.line_out = e.line_out.map(lineMap).fillna("")

# Swap line numbers
e["tmp1"] = e.line_in
e["tmp2"] = e.line_out
e.line_out = e.tmp1
e.line_in = e.tmp2

# Group nodes
n, e = lvd.ne_groupnodes(n, e)

# Reverse DDG edges for method declaration
alt_e = e[(e.line_out == 1) & (e.dataflow != "")].copy()
alt_e.outnode = alt_e.innode
alt_e.innode = 1
e = e[e.line_out != 1]
e = e.append(alt_e)

# Plot graph
n["node_label"] = n["lineNumber"].astype(str)
e = e[e.innode != e.outnode]

e.etype = e.apply(
    lambda x: f"DDG: {x.dataflow}" if len(x.dataflow) > 0 else x.etype, axis=1
)
e[e.dataflow == "!sig_none"]
en = e[e.etype != "CFG"]
en = en[en.etype != "AST"]
en = en[en.etype != "REACHING_DEF"]
en = en[en.etype != "DDG: <RET>"]
en = en[en.etype != "DDG: !sig_none"]
en = en[en.etype != "DDG: now = timespec64_to_ktime(ts64)"]
en = en.merge(n[["id", "name", "code"]], left_on="line_in", right_on="id")
# en = en[~((en.etype.str.contains("=")) & (~en.name.str.contains("assignment")))]

# Only keep assignment DDG
en.name = en.name.fillna("<operator>.assignment")
en = en[en.name == "<operator>.assignment"]
en.dataflow = en.dataflow.fillna("")
en["left_assign"] = en.code.apply(lambda x: x.split("=")[0].strip())
en = en[en.apply(lambda x: x.left_assign in x.dataflow, axis=1)]

# Add CDG edges back
en = en[(en.etype.str.contains("DDG"))]
en = en.append(e[e.etype == "CDG"])

# Add other edges
en = en.append({"innode": 3, "outnode": 18, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 3, "outnode": 22, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 3, "outnode": 25, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 4, "outnode": 9, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 4, "outnode": 19, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 4, "outnode": 25, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 5, "outnode": 18, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 5, "outnode": 19, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 6, "outnode": 8, "etype": "DDG"}, ignore_index=1)

# Reverse edges nack
en["tmp"] = en.innode
en["innode"] = en.outnode
en["outnode"] = en.tmp

n = svdj.drop_lone_nodes(n, en)
n = n.append({"id": 4, "node_label": "4"}, ignore_index=1)
dot = get_digraph(
    n[["id", "node_label"]].to_numpy().tolist(),
    en[["outnode", "innode", "etype"]].to_numpy().tolist(),
    edge_label=False,
)
dot.render("/tmp/tmp.gv", view=True)
