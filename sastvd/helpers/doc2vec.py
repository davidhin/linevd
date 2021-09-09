import logging

import sastvd.helpers.tokenise as svdt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def train_d2v(
    train_corpus,
    vector_size=300,
    window=2,
    min_count=5,
    workers=4,
    epochs=100,
    dm_concat=1,
    dm=1,
):
    """Train Doc2Vec model.

    Doc2Vec.load(savedir / "d2v.model")
    """
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    train_corpus = [
        TaggedDocument(doc.split(), [i]) for i, doc in enumerate(train_corpus)
    ]
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        dm_concat=dm_concat,
        dm=dm,
    )
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def load_d2v(path: str):
    """Load Doc2Vec model.

    path = svd.processed_dir() / "bigvul/d2v_False"
    """
    path = str(path)
    if path.split("/")[-1] != "d2v.model":
        path += "/d2v.model"
    return Doc2Vec.load(path)


class D2V:
    """Doc2Vec class."""

    def __init__(self, path: str):
        """Init class."""
        self.model = load_d2v(path)

    def infer(self, text: str):
        """Infer vector."""
        text = svdt.tokenise(text)
        return self.model.infer_vector(text.split())
