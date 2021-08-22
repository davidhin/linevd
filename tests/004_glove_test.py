import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.glove as svdg


def test_glove_bigvul_1():
    """Generate glove embeddings on small subset of BigVul."""
    svdd.generate_glove("bigvul", sample=True)


def test_glove_bigvul_2():
    """Load glove embeddings."""
    path = svd.processed_dir() / "bigvul/glove_True/vectors.txt"
    _, corp = svdg.glove_dict(path, cache=False)
    assert corp["if"] <= 10
    assert corp["ps"] <= 10
    assert corp["return"] <= 10
    assert corp["len"] <= 10


def test_glove_bigvul_3():
    """Test closest embeddings."""
    path = svd.processed_dir() / "bigvul/glove_True/vectors.txt"
    gdict, _ = svdg.glove_dict(path, cache=False)
    if_closest = svdg.find_closest_embeddings("if", gdict)
    assert if_closest[0] == "if"
    assert if_closest[1] == "else"


def test_glove_bigvul_4():
    """Test get embeddings."""
    path = svd.processed_dir() / "bigvul/glove_True/vectors.txt"
    gdict, _ = svdg.glove_dict(path, cache=False)
    ret = svdg.get_embeddings("if outofvocabword", gdict, 100)
    assert any([i != 0.01 for i in ret[0]])
    assert all([i == 0.001 for i in ret[1]])
