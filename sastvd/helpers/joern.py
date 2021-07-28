import json
import os
import pickle as pkl
import random
from pathlib import Path

import pandas as pd
import sastvd as svd
from graphviz import Digraph


def nodelabel2line(label: str):
    return label.split(":")[0].split("_")[-1]


def randcolor():
    r = lambda: random.randint(0, 255)
    return "#%02X%02X%02X" % (r(), r(), r())


def get_digraph(nodes, edges):
    """Plote digraph given nodes and edges list."""
    dot = Digraph(comment="Combined PDG")

    nodes = [n + [nodelabel2line(n[1])] for n in nodes]
    colormap = {"": "white"}
    for n in nodes:
        if n[2] not in colormap:
            colormap[n[2]] = randcolor()

    for n in nodes:
        style = {"style": "filled", "fillcolor": colormap[n[2]]}
        dot.node(str(n[0]), n[1], **style)
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
            style["color"] = "blue"
        elif e[2] == "REACHING_DEF":
            style["style"] = "solid"
            style["color"] = "orange"
        elif "DDG" in e[2]:
            style["style"] = "dashed"
            style["color"] = "darkgreen"
        else:
            style["style"] = "solid"
            style["color"] = "black"
        style["penwidth"] = "1"
        dot.edge(str(e[0]), str(e[1]), e[2], **style)
    return dot


def run_joern(filepath: str):
    """Extract graph using most recent Joern."""
    outdir = Path(filepath).parent
    # if os.path.exists(outdir / f"{Path(filepath).name}.graph.pkl"):
    # return
    script_file = svd.external_dir() / "get_func_graph.scala"
    filename = svd.external_dir() / filepath
    params = f"filename={filename}"
    svd.subprocess_cmd(
        f"joern --script {script_file} --params {params}",
        verbose=1,
    )


def get_node_edges(filepath: str):
    """Get node and edges given filepath (must run after run_joern)."""
    outdir = Path(filepath).parent
    outfile = outdir / Path(filepath).name

    with open(str(outfile) + ".edges.json", "r") as f:
        edges = json.load(f)
        edges = pd.DataFrame(edges, columns=["outnode", "innode", "etype", "dataflow"])
        edges = edges.fillna("")

    with open(str(outfile) + ".nodes.json", "r") as f:
        nodes = json.load(f)
        nodes = pd.DataFrame.from_records(nodes)
        if "controlStructureType" not in nodes.columns:
            nodes["controlStructureType"] = ""
        nodes = nodes.fillna("")
        nodes = nodes[
            ["id", "_label", "name", "code", "lineNumber", "controlStructureType"]
        ]

    # Assign node name to node code if code is null
    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x["name"], axis=1)

    # Assign node label for printing in the graph
    nodes["node_label"] = (
        nodes._label + "_" + nodes.lineNumber.astype(str) + ": " + nodes.code
    )

    # Filter by node type
    nodes = nodes[nodes._label != "COMMENT"]
    nodes = nodes[nodes._label != "FILE"]

    # Filter by edge type
    edges = edges[edges.etype != "CONTAINS"]
    edges = edges[edges.etype != "SOURCE_FILE"]
    edges = edges[edges.etype != "DOMINATE"]
    edges = edges[edges.etype != "POST_DOMINATE"]

    # Remove nodes not connected to line number nodes (maybe not efficient)
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
        left_on="outnode",
        right_on="id",
    )
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
        left_on="innode",
        right_on="id",
    )
    edges = edges[(edges.line_out != "") | (edges.line_in != "")]

    # Uniquify types
    edges.outnode = edges.apply(
        lambda x: f"{x.outnode}_{x.innode}" if x.line_out == "" else x.outnode, axis=1
    )
    typemap = nodes[["id", "name"]].set_index("id").to_dict()["name"]

    linemap = nodes.set_index("id").to_dict()["lineNumber"]
    for e in edges.itertuples():
        if type(e.outnode) == str:
            lineNum = linemap[e.innode]
            node_label = f"TYPE_{lineNum}: {typemap[int(e.outnode.split('_')[0])]}"
            nodes = nodes.append(
                {"id": e.outnode, "node_label": node_label, "lineNumber": lineNum},
                ignore_index=True,
            )

    # Save data
    with open(outdir / f"{Path(filepath).name}.graph.pkl", "wb") as f:
        pkl.dump({"nodes": nodes, "edges": edges}, f)

    return nodes, edges


def plot_node_edges(filepath: str, lineNumber: int = -1, filter_edges=[]):
    """Plot node edges given filepath (must run after get_node_edges)."""
    nodes, edges = get_node_edges(filepath)

    if len(filter_edges) > 0:
        edges = edges[edges.etype.isin(filter_edges)]

    # Draw graph
    if lineNumber > 0:
        nodesforline = set(nodes[nodes.lineNumber == lineNumber].id.tolist())
    else:
        nodesforline = set(nodes.id.tolist())

    edges_new = edges[
        (edges.outnode.isin(nodesforline)) | (edges.innode.isin(nodesforline))
    ]
    nodes_new = nodes[
        nodes.id.isin(set(edges_new.outnode.tolist() + edges_new.innode.tolist()))
    ]
    dot = get_digraph(
        nodes_new[["id", "node_label"]].to_numpy().tolist(),
        edges_new[["outnode", "innode", "etype"]].to_numpy().tolist(),
    )
    dot.render("/tmp/tmp.gv", view=True)


def full_run_joern(filepath: str):
    """Run full Joern extraction and save output."""
    try:
        run_joern(filepath)
        get_node_edges(filepath)
    except Exception as E:
        svd.debug(f"Failed {filepath}: {E}")


def full_run_joern_from_string(code: str, dataset: str, iid: str):
    """Run full joern from a string instead of file."""
    savedir = svd.get_dir(svd.interim_dir() / dataset)
    savepath = savedir / f"{iid}.c"
    with open(savepath, "w") as f:
        f.write(code)
    full_run_joern(savepath)
