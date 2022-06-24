import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from code_gnn.models.flow_gnn.gin import FlowGNNModule
from sastvd.linevd import BigVulDatasetLineVDDataModule
from pytorch_lightning.utilities.warnings import PossibleUserWarning


import warnings

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--my_verbose", action="store_true", help="Whether to use verbose logging")
        parser.link_arguments("my_verbose", "data.verbose")
        parser.link_arguments("data.feat", "model.feat")
        parser.link_arguments("data.input_dim", "model.input_dim", apply_on="instantiate")
    
    def before_instantiate_classes(self):
        if not self.config["fit.my_verbose"]:
            warnings.filterwarnings("ignore", category=PossibleUserWarning)

if __name__ == "__main__":
    cli = MyLightningCLI(FlowGNNModule, BigVulDatasetLineVDDataModule)
