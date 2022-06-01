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
from code_gnn.models.embedding_ids import node_type_map
from code_gnn.models.flow_gnn.ginconv import MyGINConv
from code_gnn.models.flow_gnn.mlp import MLP
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling

logger = logging.getLogger(__name__)

# feature_keys = {
#     "feature": "h",
#     "node_type": "node_type",
# }
feature_keys = {
    "feature": "_ABS_DATAFLOW",
    # "feature": "_1G_DATAFLOW",
    "node_type": "node_type",
}

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
    def __init__(self, input_dim, num_layers, num_mlp_layers, hidden_dim,
                 final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        output_dim = 1
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # construct neural network layers

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                MyGINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        additional_element_size = 0
        if self.hparams.node_type_separate:
            additional_element_size = len(node_type_map)

        if self.hparams.label_style == 'node':
            for layer in range(num_layers):
                i = hidden_dim
                o = hidden_dim
                if layer == 0:
                    i = hidden_dim + additional_element_size
                if layer == num_layers - 1:
                    o = output_dim
                self.linears_prediction.append(
                    nn.Linear(i, o))
        else:
            for layer in range(num_layers):
                if layer == 0:
                    self.linears_prediction.append(
                        nn.Linear(input_dim + additional_element_size, output_dim))
                else:
                    self.linears_prediction.append(
                        nn.Linear(hidden_dim + additional_element_size, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FlowGNN arguments")
        parser.add_argument("--num_layers", type=int, default=5, help='number of GIN layers to use')
        parser.add_argument("--num_mlp_layers", type=int, default=2,
                            help='number of layers to use in each GIN layer\'s MLP')
        parser.add_argument("--hidden_dim", type=int, default=32, help='width of the GIN hidden layers')
        parser.add_argument("--learn_eps", type=bool, default=False,
                            help='whether or not to learn a weight for the epsilon value')
        parser.add_argument("--final_dropout", type=float, default=0.5,
                            help='probability to use for the final dropout layer')
        parser.add_argument("--graph_pooling_type", type=str, default='sum', help='GIN graph pooling operator to use')
        parser.add_argument("--neighbor_pooling_type", type=str, default='sum', choices=all_aggregate_functions,
                            help='GIN neighbor pooling operator to use')
        parser.add_argument("--node_type_separate", action='store_true',
                            help='attach node type separately from data flow features')
        return parent_parser

    def reset_parameters(self):
        """TODO"""
        pass

    def forward(self, g):
        h = g.ndata[feature_keys["feature"]]
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        if self.hparams.label_style == 'node':
            # GIN paper page 3:
            # For node classification, the node representation h_v^K
            # of the final iteration is used for prediction.
            result = hidden_rep[-1]
            if self.hparams.node_type_separate:
                # if result.shape[0] != g.ndata['node_type'].shape[0]:
                #     logger.debug(f'{result.shape=} {result=}')
                #     logger.debug(f"{g.ndata['node_type'].shape=} {g.ndata['node_type']=}")
                result = torch.cat((result, g.ndata[feature_keys["node_type"]]), dim=1)
            for fc in self.linears_prediction:
                result = fc(result)
            result = torch.sigmoid(result).squeeze(dim=-1)
        else:
            score_over_layer = 0

            # perform pooling over all nodes in each graph in every layer
            for i, h in enumerate(hidden_rep):
                # logger.info(f'{i} shape={h.shape} h={h}')
                if self.hparams.node_type_separate:
                    logger.warning('NOT WORKING, UNDER CONSTRUCTION...')
                    # if h.shape[0] != g.ndata['node_type'].shape[0]:
                    #     logger.debug(f'{h.shape=} {h=}')
                    #     logger.debug(f"{g.ndata['node_type'].shape=} {g.ndata['node_type']=}")
                    h = torch.cat((h, g.ndata[feature_keys["node_type"]]), dim=1)
                    # TODO: we want to pass this through a linear layer so that the one-hot gets picked up.
                h = self.pool(g, h)
                fc_out = self.linears_prediction[i](h)
                drop = self.drop(fc_out)
                # logger.info(f"drop shape={drop.shape} drop={drop} score_over_layer={score_over_layer}")
                score_over_layer += drop

            result = torch.sigmoid(score_over_layer).squeeze(dim=-1)

        return result
