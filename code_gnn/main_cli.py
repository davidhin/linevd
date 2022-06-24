import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from code_gnn.models.flow_gnn.gin import FlowGNNModule
from sastvd.linevd import BigVulDatasetLineVDDataModule

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.feat", "model.feat")
        parser.link_arguments("data.input_dim", "model.input_dim", apply_on="instantiate")
        pass

if __name__ == "__main__":
    cli = MyLightningCLI(FlowGNNModule, BigVulDatasetLineVDDataModule)
