import torch

from code_gnn.models.base_module import BaseModule


class BaseBaselineModule(BaseModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Single dummy parameter
        self.W = torch.nn.Parameter(torch.randn(1))
        self.W.requires_grad = True

    def reset_parameters(self):
        pass


class RandomModel(BaseBaselineModule):
    """Always returns random predictions."""

    def forward(self, batch):
        return torch.rand((batch.batch_size,), requires_grad=True).to(batch.device)


class OnesModel(BaseBaselineModule):
    """Always returns 1."""

    def forward(self, batch):
        return torch.ones((batch.batch_size,), requires_grad=True).to(batch.device)


class ZerosModel(BaseBaselineModule):
    """Always returns 0."""

    def forward(self, batch):
        return torch.zeros((batch.batch_size,), requires_grad=True).to(batch.device)
