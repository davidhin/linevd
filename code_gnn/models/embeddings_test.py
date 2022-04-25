from code_gnn.analysis.cpg import get_cpg
from code_gnn.globals import test_data_dir
from code_gnn.models.embeddings import DataflowEmbeddingGetter


def test_dataflow():
    getter = DataflowEmbeddingGetter()
    # TODO: Check parsed first
    cpg = get_cpg(test_data_dir / 'dataflow_test/parsed', 'test.c')
    print(getter.get_embedding(cpg, None))
