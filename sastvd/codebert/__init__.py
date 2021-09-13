import os

import matplotlib.pyplot as plt
import sastvd as svd
import torch
from transformers import AutoModel, AutoTokenizer
from tsne_torch import TorchTSNE as TSNE


class CodeBert:
    """CodeBert.

    Example:
    cb = CodeBert()
    sent = ["int myfunciscool(float b) { return 1; }", "int main"]
    ret = cb.encode(sent)
    ret.shape
    >>> torch.Size([2, 768])
    """

    def __init__(self):
        """Initiate model."""
        codebert_base_path = svd.external_dir() / "codebert-base"
        if os.path.exists(codebert_base_path):
            self.tokenizer = AutoTokenizer.from_pretrained(codebert_base_path)
            self.model = AutoModel.from_pretrained(codebert_base_path)
        else:
            cache_dir = svd.get_dir(svd.cache_dir() / "codebert_model")
            print("Loading Codebert...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base", cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                "microsoft/codebert-base", cache_dir=cache_dir
            )
        self._dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self._dev)

    def encode(self, sents: list):
        """Get CodeBert embeddings from a list of sentences."""
        tokens = [i for i in sents]
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer(tokens, **tk_args).to(self._dev)
        with torch.no_grad():
            return self.model(tokens["input_ids"], tokens["attention_mask"])[1]


def plot_embeddings(embeddings, words):
    """Plot embeddings.

    import sastvd.helpers.datasets as svdd
    cb = CodeBert()
    df = svdd.bigvul()
    sent = " ".join(df.sample(5).before.tolist()).split()
    plot_embeddings(cb.encode(sent), sent)
    """
    tsne = TSNE(n_components=2, n_iter=2000, verbose=True)
    Y = tsne.fit_transform(embeddings)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(words, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()
