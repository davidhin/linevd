import abc
import logging

import dgl
import gensim
import networkx as nx
import torch

from code_gnn.analysis.dataflow import ReachingDefinitions
from code_gnn.analysis.nx_utils import normalize
from code_gnn.globals import wv_cache_path
from code_gnn.models.embedding_ids import type_one_hot

logger = logging.getLogger(__name__)


def is_valid_for_dgl(graph):
    if list(sorted(graph.nodes)) != list(range(0, len(graph.nodes))):
        logger.error("First node's index must start from 0 and node indices must be sequential. DGL rules maaaaan.")
        return False
    if len(graph.nodes) == 0:
        logger.error(f'Empty graph after embedding')
        return False
    return True


class EmbeddingGetter(abc.ABC):
    def get_graph(self, cpg):
        pass

    def __repr__(self):
        return f'{type(self).__name__}'


class NoFeaturesEmbeddingGetter(EmbeddingGetter):
    def __init__(self):
        pass

    def get_graph(self, cpg):
        cpg = normalize(cpg)
        assert is_valid_for_dgl(cpg)
        us, vs = zip(*cpg.edges(data=False))
        g = dgl.graph((us, vs))
        # g.ndata['h'] = torch.rand((g.number_of_nodes(), 1))
        return g


class DevignEmbeddingGetter(EmbeddingGetter):
    def __init__(self, dataset_name, window_size, df):
        self.window_size = window_size

        # initialize Word2Vec
        all_sentences = df["code"].apply(lambda f: f.split())
        window = 5
        min_count = 1
        workers = 4
        wv_cache_file = wv_cache_path / ('word2vec_model_' + '_'.join(
            map(str, [dataset_name, self.window_size, window, min_count, workers])) + '.bin')
        force = False  # force retrain/save
        if not force and wv_cache_file.exists():
            self.wv_model = gensim.models.Word2Vec.load(str(wv_cache_file))
        else:
            self.wv_model = gensim.models.Word2Vec(
                sentences=all_sentences,
                vector_size=self.window_size,
                window=window,
                min_count=min_count,
                workers=workers)
            self.wv_model.save(str(wv_cache_file))

    def __repr__(self):
        return f'{type(self).__name__}({self.window_size=})'

    # python code_gnn/main.py --model flow_gnn --dataset MSR --clean --batch_size 256 --max_epochs 100 --undersample_factor 1.0 --label_style node --node_type_separate --filter npd+undersample_unrealistic --take_checkpoint last --neighbor_pooling_type sum
    def get_graph(self, cpg):
        node_code = nx.get_node_attributes(cpg, 'code')

        def get_node_embedding(node):
            code = node_code[node]
            tokens = code.strip().split()
            token_embeddings = [torch.tensor(self.wv_model.wv[tok]) for tok in tokens if
                                tok in self.wv_model.wv]
            if len(token_embeddings) == 0:
                node_embedding = torch.zeros(self.wv_model.vector_size)
            else:
                node_embedding = torch.stack(token_embeddings, dim=0).mean(dim=0)
            return node_embedding

        w2v_node_embeddings = [get_node_embedding(node) for node in cpg.nodes]
        w2v_graph_embedding = torch.stack(w2v_node_embeddings, dim=0)

        node_type = nx.get_node_attributes(cpg, 'type')
        type_embeddings = torch.stack([type_one_hot[node_type[node] - 1] for node in cpg.nodes])

        node_label = torch.LongTensor([l for _, l in cpg.nodes(data='node_label')])
        graph_label = torch.LongTensor([l for _, l in cpg.nodes(data='graph_label')])

        cpg = normalize(cpg)
        assert is_valid_for_dgl(cpg)

        us, vs, edge_types = zip(*cpg.edges(data='type'))
        etype = torch.tensor(edge_types)

        g = dgl.graph((us, vs))
        # TODO: Currently, node type is one-hot and edge type is numeric.
        #  Experiment with numeric vs. one-hot embedding.
        g.ndata['h'] = torch.cat((type_embeddings, w2v_graph_embedding), dim=1)
        g.edata['etype'] = etype
        g.ndata['node_label'] = node_label
        g.ndata['graph_label'] = graph_label

        return g


class DataflowEmbeddingGetter(EmbeddingGetter):
    def __init__(self, cpgs, node_type_separate, max_width=None):
        self.node_type_separate = node_type_separate
        if max_width is not None:
            self.max_width = max_width
        else:
            self.max_width = max(len(ReachingDefinitions(cpg).variables) if cpg is not None else -1 for cpg in cpgs)
        # self.max_width = None

    def __repr__(self):
        return f'{type(self).__name__}({self.max_width=})'

    def get_graph(self, cpg):
        problem = ReachingDefinitions(cpg)
        graph = problem.cfg
        variables = list(sorted(problem.variables))
        dataflow_embeddings = torch.zeros((len(graph.nodes), len(variables)), dtype=torch.int)
        for i, node in enumerate(graph.nodes):
            gen = problem.gen(node)
            if len(gen) > 0:
                for variable, _ in gen:
                    dataflow_embeddings[i][variables.index(variable)] = 1
        dataflow_embeddings = torch.zeros((dataflow_embeddings.size(0), self.max_width))  # Assume 2d
        # dataflow_embeddings_pad[:, :dataflow_embeddings.size(1)] = dataflow_embeddings
        # dataflow_embeddings = dataflow_embeddings_pad
        node_embeddings = dataflow_embeddings

        node_type = nx.get_node_attributes(graph, 'type')
        type_embeddings = torch.stack([type_one_hot[node_type[node] - 1] for node in graph.nodes])

        node_label = torch.LongTensor([l for _, l in graph.nodes(data='node_label')])
        graph_label = torch.LongTensor([l for _, l in graph.nodes(data='graph_label')])

        graph = normalize(graph)
        assert is_valid_for_dgl(graph)
        # TODO: Move this normalization to CPG extraction stage.
        # TODO: Store embeddings in separate files and use simple method to combine DGL graph.

        # Get DGL graph
        us, vs = zip(*graph.edges())
        g = dgl.graph((us, vs))
        g.ndata['h'] = node_embeddings
        g.ndata['node_type'] = type_embeddings
        g.ndata['node_label'] = node_label
        g.ndata['graph_label'] = graph_label
        return g
