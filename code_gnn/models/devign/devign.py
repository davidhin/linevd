import dgl
import torch
from dgl.nn.pytorch import GatedGraphConv
from torch import nn
from torch.nn import functional as F

from code_gnn.models.base_module import BaseModule
from code_gnn.models.embedding_ids import edge_type_map


def stack_pad_zeros(graph_features):
    features = []
    max_length = max(len(g) for g in graph_features)
    for feature in graph_features:
        pad_dim = max_length - len(feature)
        feature = torch.cat(
            (feature,
             torch.zeros(
                 size=(pad_dim, *(feature.shape[1:])),
                 requires_grad=feature.requires_grad,
                 device=feature.device)),
            dim=0)
        features.append(feature)
    return torch.stack(features, dim=0)


class DevignModule(BaseModule):
    def __init__(self, input_dim, window_size, graph_embed_size, num_layers, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        n_etypes = max(edge_type_map.values()) + 1
        total_input_dim = input_dim
        if graph_embed_size >= total_input_dim:
            ggnn_input_dim = total_input_dim
            self.downsample_layer = None
        else:
            ggnn_input_dim = graph_embed_size
            print(f'Creating downsample layer from {total_input_dim} to {ggnn_input_dim}')
            self.downsample_layer = torch.nn.Linear(total_input_dim, ggnn_input_dim)

        # construct neural network layers
        self.ggnn = GatedGraphConv(in_feats=ggnn_input_dim, out_feats=graph_embed_size,
                                   n_steps=num_layers, n_etypes=n_etypes)
        self.conv_l1 = torch.nn.Conv1d(graph_embed_size, graph_embed_size, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(graph_embed_size, graph_embed_size, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = total_input_dim + graph_embed_size
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=graph_embed_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Devign arguments")
        parser.add_argument("--window_size", type=int, default=100, help='(should be renamed) vector size for Word2Vec')
        parser.add_argument("--graph_embed_size", type=int, default=169, help='width of the GNN node embedding')
        parser.add_argument("--num_layers", type=int, default=6, help='number of GNN layers')
        return parent_parser

    def reset_parameters(self):
        self.ggnn.reset_parameters()
        self.conv_l1.reset_parameters()
        self.conv_l2.reset_parameters()
        self.conv_l1_for_concat.reset_parameters()
        self.conv_l2_for_concat.reset_parameters()
        self.mlp_z.reset_parameters()
        self.mlp_y.reset_parameters()

    def forward(self, batch):
        features = batch.ndata['h']
        if self.downsample_layer is not None:
            features = self.downsample_layer(features)
        batch.ndata['h_next'] = self.ggnn(batch, features, batch.edata['etype'])
        graphs = dgl.unbatch(batch, batch.batch_num_nodes(), batch.batch_num_edges())
        features_pad = stack_pad_zeros([g.ndata['h'] for g in graphs])
        h_i = stack_pad_zeros([g.ndata['h_next'] for g in graphs])
        c_i = torch.cat((h_i, features_pad), dim=-1)
        Y_1 = self.maxpool1(
            F.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            F.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            F.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            F.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result
