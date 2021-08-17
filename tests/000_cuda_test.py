import torch


def test_cuda():
    """Test if CUDA is available."""
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert str(dev) == "cuda:0"
