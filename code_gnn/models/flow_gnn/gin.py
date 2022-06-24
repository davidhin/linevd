"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from code_gnn.globals import all_aggregate_functions
from code_gnn.models.base_module import BaseModule
from code_gnn.models.flow_gnn.ginconv import MyGINConv
from code_gnn.models.flow_gnn.mlp import MLP
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling

logger = logging.getLogger(__name__)

# feature_keys = {
#     "feature": "h",
#     "node_type": "node_type",
# }
"""
feature_keys = {
    "feature": "_ABS_DATAFLOW",
    # "feature": "_1G_DATAFLOW",
    "node_type": "node_type",
}
"""


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class FlowGNNModule(BaseModule):
    def __init__(
        self,
        feat,
        input_dim,
        label_style="graph",
        num_layers=5,
        num_mlp_layers=2,
        hidden_dim=32,
        final_dropout=0.5,
        learn_eps=False,
        graph_pooling_type="sum",
        neighbor_pooling_type="sum",
        separate_embedding_layer=False,
        node_type_separate=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.construct_model(feat, input_dim, label_style, num_layers, num_mlp_layers, hidden_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, separate_embedding_layer)

    def construct_model(self, feat, input_dim, label_style, num_layers, num_mlp_layers, hidden_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, separate_embedding_layer):
        output_dim = 1
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        if "_ABS_DATAFLOW" in feat:
            feat = "_ABS_DATAFLOW"
        self.feature_keys = {
            "feature": feat,
            "node_type": "node_type",
        }

        # construct neural network layers
        if separate_embedding_layer:
            self.embedding = nn.Linear(input_dim, hidden_dim)

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0 and not separate_embedding_layer:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                MyGINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        additional_element_size = 0
        # if self.hparams.node_type_separate:
        #     additional_element_size = len(node_type_map)

        if label_style == "node":
            for layer in range(num_layers):
                i = hidden_dim
                o = hidden_dim
                if layer == 0:
                    i = hidden_dim + additional_element_size
                if layer == num_layers - 1:
                    o = output_dim
                self.linears_prediction.append(nn.Linear(i, o))
        else:
            for layer in range(num_layers):
                if layer == 0 and not separate_embedding_layer:
                    self.linears_prediction.append(
                        nn.Linear(input_dim + additional_element_size, output_dim)
                    )
                else:
                    self.linears_prediction.append(
                        nn.Linear(hidden_dim + additional_element_size, output_dim)
                    )

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == "sum":
            self.pool = SumPooling()
        elif graph_pooling_type == "mean":
            self.pool = AvgPooling()
        elif graph_pooling_type == "max":
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def reset_parameters(self):
        """TODO"""
        pass

    def forward(self, g):
        h = g.ndata[self.feature_keys["feature"]]

        if self.hparams.separate_embedding_layer:
            h = self.embedding(h)

        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        if self.hparams.label_style == "node":
            # GIN paper page 3:
            # For node classification, the node representation h_v^K
            # of the final iteration is used for prediction.
            result = hidden_rep[-1]
            if self.hparams.node_type_separate:
                # if result.shape[0] != g.ndata['node_type'].shape[0]:
                #     logger.debug(f'{result.shape=} {result=}')
                #     logger.debug(f"{g.ndata['node_type'].shape=} {g.ndata['node_type']=}")
                result = torch.cat((result, g.ndata[self.feature_keys["node_type"]]), dim=1)
            for fc in self.linears_prediction:
                result = fc(result)
            result = torch.sigmoid(result).squeeze(dim=-1)
        else:
            score_over_layer = 0

            # perform pooling over all nodes in each graph in every layer
            for i, h in enumerate(hidden_rep):
                # logger.info(f'{i} shape={h.shape} h={h}')
                if self.hparams.node_type_separate:
                    logger.warning("NOT WORKING, UNDER CONSTRUCTION...")
                    # if h.shape[0] != g.ndata['node_type'].shape[0]:
                    #     logger.debug(f'{h.shape=} {h=}')
                    #     logger.debug(f"{g.ndata['node_type'].shape=} {g.ndata['node_type']=}")
                    h = torch.cat((h, g.ndata[self.feature_keys["node_type"]]), dim=1)
                    # TODO: we want to pass this through a linear layer so that the one-hot gets picked up.
                h = self.pool(g, h)
                fc_out = self.linears_prediction[i](h)
                drop = self.drop(fc_out)
                # logger.info(f"drop shape={drop.shape} drop={drop} score_over_layer={score_over_layer}")
                score_over_layer += drop

            result = torch.sigmoid(score_over_layer).squeeze(dim=-1)

        return result
