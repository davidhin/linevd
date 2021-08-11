"""Modified from https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py."""

import warnings

import dgl
import torch as th
import torch.nn as nn

# This warning also appears in official DGL Tree-LSTM docs, so ignore it.
warnings.filterwarnings("ignore", message="The input graph for the user-defined edge")


class ChildSumTreeLSTMCell(nn.Module):
    """Copied from official implementation."""

    def __init__(self, x_size, h_size):
        """Init."""
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        """Message UDF."""
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        """Reduce UDF."""
        h_tild = th.sum(nodes.mailbox["h"], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox["h"]))
        c = th.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_tild), "c": c}

    def apply_node_func(self, nodes):
        """Apply UDF."""
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * th.tanh(c)
        return {"h": h, "c": c}


class TreeLSTM(nn.Module):
    """Customised N-ary TreeLSTM.

    Example:
    a = BigVulGraphDataset(sample=10)
    asts = a.item(180189)["asts"]
    batched_g = dgl.batch([i for i in asts if i])
    model = TreeLSTM(200, 200)
    model(batched_g)
    """

    def __init__(
        self,
        x_size,
        h_size,
        dropout=0,
    ):
        """Init.

        Args:
            x_size (int): Input size.
            h_size (int): Hidden size.
            dropout (int): Dropout (final layer)
        """
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.cell = ChildSumTreeLSTMCell(x_size, h_size)
        self.h_size = h_size
        self.dev = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    def forward(self, g):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        g : dgl.DGLGraph
            Tree for computation.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # feed embedding
        embeds = g.ndata["_FEAT"].to(self.dev)
        n = g.number_of_nodes()
        g.ndata["iou"] = self.cell.W_iou(self.dropout(embeds))
        g.ndata["h"] = th.zeros((n, self.h_size)).to(self.dev)
        g.ndata["c"] = th.zeros((n, self.h_size)).to(self.dev)

        # propagate
        dgl.prop_nodes_topo(
            g,
            self.cell.message_func,
            self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )
        # return hidden state
        h = self.dropout(g.ndata.pop("h"))

        # unbatch and get root node (ASSUME ROOT NODE AT IDX=0)
        g.ndata["h"] = h
        unbatched = dgl.unbatch(g)
        return dict(
            [
                [
                    (i.ndata["_ID"][0].int().item(), i.ndata["_LINE"][0].int().item()),
                    i.ndata["h"][0],
                ]
                for i in unbatched
            ]
        )
