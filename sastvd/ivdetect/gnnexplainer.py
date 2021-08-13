import math

import dgl
import dgl.function as fn
import matplotlib.pylab as plt
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class NodeExplainerModule(nn.Module):
    """
    A Pytorch module for explaining a node's prediction based on its computational graph and node features.
    Use two masks: One mask on edges, and another on nodes' features.
    So far due to the limit of DGL on edge mask operation, this explainer need the to-be-explained models to
    accept an additional input argument, edge mask, and apply this mask in their inner message parse operation.
    This is current walk_around to use edge masks.
    """

    # Class inner variables
    loss_coef = {"g_size": 0.05, "feat_size": 1.0, "g_ent": 0.1, "feat_ent": 0.1}

    def __init__(
        self,
        model,
        num_edges,
        node_feat_dim,
        activation="sigmoid",
        agg_fn="sum",
        mask_bias=False,
    ):
        super(NodeExplainerModule, self).__init__()
        self.model = model
        self.model.eval()
        self.num_edges = num_edges
        self.node_feat_dim = node_feat_dim
        self.activation = activation
        self.agg_fn = agg_fn
        self.mask_bias = mask_bias

        # Initialize parameters on masks
        self.edge_mask, self.edge_mask_bias = self.create_edge_mask(self.num_edges)
        self.node_feat_mask = self.create_node_feat_mask(self.node_feat_dim)

    def create_edge_mask(self, num_edges, init_strategy="normal", const=1.0):
        """
        Based on the number of nodes in the computational graph, create a learnable mask of edges.
        To adopt to DGL, change this mask from N*N adjacency matrix to the No. of edges
        Parameters
        ----------
        num_edges: Integer N, specify the number of edges.
        init_strategy: String, specify the parameter initialization method
        const: Float, a value for constant initialization
        Returns
        -------
        mask and mask bias: Tensor, all in shape of N*1
        """
        mask = nn.Parameter(th.Tensor(num_edges, 1))

        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(1.0 / num_edges)
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const)

        if self.mask_bias:
            mask_bias = nn.Parameter(th.Tensor(num_edges, 1))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def create_node_feat_mask(self, node_feat_dim, init_strategy="normal"):
        """
        Based on the dimensions of node feature in the computational graph, create a learnable mask of features.
        Parameters
        ----------
        node_feat_dim: Integer N, dimensions of node feature
        init_strategy: String, specify the parameter initialization method
        Returns
        -------
        mask: Tensor, in shape of N
        """
        mask = nn.Parameter(th.Tensor(node_feat_dim))

        if init_strategy == "normal":
            std = 0.1
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with th.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def forward(self, graph, n_feats, dataset):
        """
        Calculate prediction results after masking input of the given model.
        Parameters
        ----------
        graph: DGLGraph, Should be a sub_graph of the target node to be explained.
        n_idx: Tensor, an integer, index of the node to be explained.
        Returns
        -------
        new_logits: Tensor, in shape of N * Num_Classes
        """

        # Step 1: Mask node feature with the inner feature mask
        new_n_feats = n_feats * self.node_feat_mask.sigmoid()
        edge_mask = self.edge_mask.sigmoid()

        # Step 2: Add compute logits after mask node features and edges
        graph.ndata["_FEAT"] = new_n_feats
        new_logits = self.model(graph, dataset, edge_mask)

        return new_logits

    def _loss(self, pred_logits, pred_label):
        """
        Compute the losses of this explainer, which include 6 parts in author's codes:
        1. The prediction loss between predict logits before and after node and edge masking;
        2. Loss of edge mask itself, which tries to put the mask value to either 0 or 1;
        3. Loss of node feature mask itself,  which tries to put the mask value to either 0 or 1;
        4. L2 loss of edge mask weights, but in sum not in mean;
        5. L2 loss of node feature mask weights, which is NOT used in the author's codes;
        6. Laplacian loss of the adj matrix.
        In the PyG implementation, there are 5 types of losses:
        1. The prediction loss between logits before and after node and edge masking;
        2. Sum loss of edge mask weights;
        3. Loss of edge mask entropy, which tries to put the mask value to either 0 or 1;
        4. Sum loss of node feature mask weights;
        5. Loss of node feature mask entropy, which tries to put the mask value to either 0 or 1;
        Parameters
        ----------
        pred_logits：Tensor, N-dim logits output of model
        pred_label: Tensor, N-dim one-hot label of the label
        Returns
        -------
        loss: Scalar, the overall loss of this explainer.
        """
        # 1. prediction loss
        log_logit = -F.log_softmax(pred_logits, dim=-1)
        pred_loss = th.sum(log_logit * pred_label)

        # 2. edge mask loss
        if self.activation == "sigmoid":
            edge_mask = th.sigmoid(self.edge_mask)
        elif self.activation == "relu":
            edge_mask = F.relu(self.edge_mask)
        else:
            raise ValueError()
        edge_mask_loss = self.loss_coef["g_size"] * th.sum(edge_mask)

        # 3. edge mask entropy loss
        edge_ent = -edge_mask * th.log(edge_mask + 1e-8) - (1 - edge_mask) * th.log(
            1 - edge_mask + 1e-8
        )
        edge_ent_loss = self.loss_coef["g_ent"] * th.mean(edge_ent)

        # 4. node feature mask loss
        if self.activation == "sigmoid":
            node_feat_mask = th.sigmoid(self.node_feat_mask)
        elif self.activation == "relu":
            node_feat_mask = F.relu(self.node_feat_mask)
        else:
            raise ValueError()
        node_feat_mask_loss = self.loss_coef["feat_size"] * th.sum(node_feat_mask)

        # 5. node feature mask entry loss
        node_feat_ent = -node_feat_mask * th.log(node_feat_mask + 1e-8) - (
            1 - node_feat_mask
        ) * th.log(1 - node_feat_mask + 1e-8)
        node_feat_ent_loss = self.loss_coef["feat_ent"] * th.mean(node_feat_ent)

        total_loss = (
            pred_loss
            + edge_mask_loss
            + edge_ent_loss
            + node_feat_mask_loss
            + node_feat_ent_loss
        )

        return total_loss


def visualize_sub_graph(
    sub_graph, edge_weights=None, origin_nodes=None, center_node=None
):
    """
    Use networkx to visualize the sub_graph and,
    if edge weights are given, set edges with different fading of blue.
    Parameters
    ----------
    sub_graph: DGLGraph, the sub_graph to be visualized.
    edge_weights: Tensor, the same number of edges. Values are (0,1), default is None
    origin_nodes: List, list of node ids that will be used to replace the node ids in the subgraph in visualization
    center_node: Tensor, the node id in origin node list to be highlighted with different color
    Returns
    show the sub_graph
    -------
    """
    # Extract original idx and map to the new networkx graph
    # Convert to networkx graph
    g = dgl.to_networkx(sub_graph)
    nx_edges = g.edges(data=True)

    if not (origin_nodes is None):
        n_mapping = {
            new_id: old_id for new_id, old_id in enumerate(origin_nodes.tolist())
        }
        g = nx.relabel_nodes(g, mapping=n_mapping)

    pos = nx.spring_layout(g)

    if edge_weights is None:
        options = {
            "node_size": 1000,
            "alpha": 0.9,
            "font_size": 24,
            "width": 4,
        }
    else:

        ec = [edge_weights[e[2]["id"]][0] for e in nx_edges]
        options = {
            "node_size": 1000,
            "alpha": 0.3,
            "font_size": 12,
            "edge_color": ec,
            "width": 4,
            "edge_cmap": plt.cm.Reds,
            "edge_vmin": 0,
            "edge_vmax": 1,
            "connectionstyle": "arc3,rad=0.1",
        }

    nx.draw(g, pos, with_labels=True, node_color="b", **options)
    if not (center_node is None):
        nx.draw(
            g,
            pos,
            nodelist=center_node.tolist(),
            with_labels=True,
            node_color="r",
            **options
        )

    plt.show()


def gnnexplainer(model, graph, dataset):
    """Run GNNExplainer to obtain lines in ranked importance."""
    dev = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    # Create an explainer
    explainer = NodeExplainerModule(
        model=model, num_edges=graph.number_of_edges(), node_feat_dim=200
    )
    explainer.to(dev)

    # define optimizer
    optim = th.optim.Adam(explainer.parameters(), lr=0.01, weight_decay=0)

    # train the explainer for the given node
    model.eval()
    model_logits = model(graph, dataset)
    model_predict = F.one_hot(th.argmax(model_logits, dim=-1), 2)
    sub_feats = graph.ndata["_FEAT"]

    for epoch in tqdm(range(50)):
        explainer.train()
        exp_logits = explainer(graph, sub_feats, dataset)
        loss = explainer._loss(exp_logits, model_predict[0])

        optim.zero_grad()
        loss.backward()
        optim.step()

    # visualize the importance of edges
    edge_weights = explainer.edge_mask.sigmoid().detach()

    # Get Aggregate weight importances into nodes
    graph.ndata["line_importance"] = th.ones(graph.number_of_nodes(), device=dev) * 2
    graph.edata["edge_mask"] = edge_weights
    graph.update_all(
        fn.u_mul_e("line_importance", "edge_mask", "m"), fn.mean("m", "line_importance")
    )

    # Return sorted list of line importances
    ret = sorted(
        list(
            zip(
                graph.ndata["line_importance"].squeeze().detach().cpu().numpy(),
                graph.ndata["_LINE"].detach().cpu().numpy(),
            )
        ),
        reverse=True,
    )

    return [i[1] for i in ret]
