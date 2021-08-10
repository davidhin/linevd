import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

mpl.rcParams["figure.dpi"] = 300


def simple_nx_plot(outnodes, innodes, node_labels):
    """Plot graph for debugging purposes."""
    labels = dict([(i, j) for i, j in enumerate(node_labels)])
    G = nx.DiGraph(
        list(zip(outnodes, innodes)),
    )
    G.add_nodes_from(range(len(node_labels)))

    options = {
        "font_size": 6,
        "node_size": 300,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 0.5,
        "width": 0.5,
        "labels": labels,
        "with_labels": True,
    }
    pos = graphviz_layout(G, prog="dot")
    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.10)
    plt.axis("off")
    plt.show()


def simple_dgl_plot(dglgraph):
    """Plot DGL graph simple."""
    G = dglgraph.to_networkx()
    options = {
        "font_size": 6,
        "node_size": 300,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 0.5,
        "width": 0.5,
    }
    pos = graphviz_layout(G, prog="dot")
    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.10)
    plt.axis("off")
    plt.show()
