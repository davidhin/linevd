import json

import numpy as np
import pytest
import pytorch_lightning
import torch
import torch_geometric
from torch.nn import BCELoss
from torch.optim import Adam

from code_gnn.dataloader import MyDataLoader
from code_gnn.dataset import MyDGLDataset
from code_gnn.models import FlowGNNModule


def test_stats():
    """
    Output test to print statistics of dataset.
    Last output (3/15/2022 10:40 PM)
    num_all_nonvuln_graphs=193
    num_graphs=38410
    num_vuln=58217
    num_nodes=674601
    percent_vuln (all)=8.63%
    percent_vuln (average)=12.50%
    """

    dataset = MyDGLDataset({
        "dataset": "SARD",
        "model": "flow_gnn",
        "node_limit": None,
        "graph_limit": None,
        "label_style": "node",
    })
    num_graphs = 0
    num_all_nonvuln_graphs = 0
    num_vuln = 0
    num_nodes = 0
    percent_vulns = []
    for g in dataset.graphs:
        num_nodes += g.number_of_nodes()
        num_vuln_in_graph = g.ndata['label'].sum().item()
        num_vuln += num_vuln_in_graph
        if num_vuln_in_graph == 0:
            num_all_nonvuln_graphs += 1
        num_graphs += 1
        percent_vulns.append(num_vuln_in_graph / g.number_of_nodes())

    print()
    print(f'{num_all_nonvuln_graphs=}')
    print(f'{num_graphs=}')
    print(f'{num_vuln=}')
    print(f'{num_nodes=}')
    print(f'percent_vuln (all)={num_vuln/num_nodes*100:.2f}%')
    print(f'percent_vuln (average)={np.average(percent_vulns)*100:.2f}%')


def test_weights():
    """
    https://github.com/suriyadeepan/torchtest/blob/66a2c8b669aa23601f64e208463e9449ffc135da/torchtest/torchtest.py#L106
    """

    pytorch_lightning.seed_everything(0)
    torch_geometric.seed_everything(0)

    dataset = MyDGLDataset({
        "dataset": "SARD",
        "model": "flow_gnn",
        "node_limit": None,
        "graph_limit": None,
        "label_style": "node",
    })
    dataloader = MyDataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=32,
    )

    model = FlowGNNModule(
        input_dim=dataset.input_dim, hidden_dim=8, num_layers=3, num_mlp_layers=3,
        final_dropout=0.5, learn_eps=True, graph_pooling_type='sum', neighbor_pooling_type='sum',
        label_style='node',
    )
    initial_params = [(name, p.clone()) for (name, p) in model.named_parameters() if p.requires_grad]

    batch_graph, _ = next(iter(dataloader))
    out = model.forward(batch_graph)
    optim = Adam(model.parameters())
    loss_fn = BCELoss()

    optim.zero_grad()
    loss = loss_fn(out, batch_graph.ndata['label'].float())
    loss.backward()
    optim.step()

    after_backward_parameters = [(name, p.clone()) for (name, p) in model.named_parameters() if p.requires_grad]

    assert loss != 0
    assert len(initial_params) == len(after_backward_parameters)
    failed = []
    for i, ((_, a), (b_name, b)) in enumerate(zip(initial_params, after_backward_parameters)):
        if torch.equal(a, b):
            failed.append((i, b_name, a, b))
    assert len(failed) == 0, f'failed indices: {failed}, values {json.dumps(failed, indent=2)}'


if __name__ == '__main__':
    pytest.main()
