from .baselines.baseline_models import RandomModel
from .flow_gnn.gin import FlowGNNModule

model_class_dict = {
    "flow_gnn": FlowGNNModule,
    "flow_gnn_only": FlowGNNModule,
    "random": RandomModel,
}
