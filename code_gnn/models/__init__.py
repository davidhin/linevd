from .baselines.baseline_models import RandomModel
from .devign.devign import DevignModule
from .flow_gnn.gin import FlowGNNModule

model_class_dict = {
    "devign": DevignModule,
    "flow_gnn": FlowGNNModule,
    "flow_gnn_only": FlowGNNModule,
    "random": RandomModel,
}
