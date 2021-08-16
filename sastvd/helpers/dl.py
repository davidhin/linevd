from fnmatch import fnmatch

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset


def tensor_memory(debug="len", verbose=1):
    """Get all tensors in memory."""
    import gc

    import torch

    ret = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                ret.append(f"{type(obj)} : {obj.device} : {obj.size()}")
        except:
            pass
    if verbose > 0:
        if debug == "len":
            print(len(ret))
        if debug == "values":
            print("\n".join(ret))
    return ret


class BatchDict:
    """Wrapper class for dicts with helper function to move attrs to torch device.

    Example:
    bd = BatchDict({"feat": torch.Tensor([1, 2, 3]), "labels": [1, 2, 3]})
    """

    def __init__(self, batch: dict):
        """Set dict as class instance attributes."""
        for k, v in batch.items():
            setattr(self, k, v)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def cuda(self, exclude: list = []):
        """Move relevant attributes to device."""
        for i in self.__dict__:
            skip = False
            for j in exclude:
                if fnmatch(i, j):
                    skip = True
            if skip:
                continue
            if hasattr(self, i):
                if isinstance(getattr(self, i), torch.Tensor):
                    setattr(self, i, getattr(self, i).to(self._device))

    def __repr__(self):
        """Override representation method."""
        return str(self.__dict__)

    def __getitem__(self, key):
        """Implement subscriptable."""
        return getattr(self, key)


class CustomDataset(Dataset):
    """Template custom dataset."""

    def __init__(self, X, y):
        """Init."""
        self.data = X
        self.target = y
        self.length = [x.count_nonzero(dim=1).count_nonzero().item() for x in X]

    def __getitem__(self, index):
        """Get item."""
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x, y, x_len

    def __len__(self):
        """Length."""
        return len(self.data)


class DynamicRNN(nn.Module):
    """Wrapper function to enable packed sequence RNNs.

    Copied from: https://gist.github.com/davidnvq/594c539b76fc52bef49ec2332e6bcd15
    """

    def __init__(self, rnn_module):
        """Init wrapper."""
        super().__init__()
        self.rnn_module = rnn_module

    def forward(self, x, len_x, initial_state=None):
        """
        Forward pass.

        Arguments
        ---------
        x : torch.FloatTensor
                padded input sequence tensor for RNN model
                Shape [batch_size, max_seq_len, embed_size]
        len_x : torch.LongTensor
                Length of sequences (b, )
        initial_state : torch.FloatTensor
                Initial (hidden, cell) states of RNN model.
        Returns
        -------
        A tuple of (padded_output, h_n) or (padded_output, (h_n, c_n))
                padded_output: torch.FloatTensor
                        The output of all hidden for each elements. The hidden of padding elements will be assigned to
                        a zero vector.
                        Shape [batch_size, max_seq_len, hidden_size]
                h_n: torch.FloatTensor
                        The hidden state of the last step for each packed sequence (not including padding elements)
                        Shape [batch_size, hidden_size]
                c_n: torch.FloatTensor
                        If rnn_model is RNN, c_n = None
                        The cell state of the last step for each packed sequence (not including padding elements)
                        Shape [batch_size, hidden_size]
        """
        # First sort the sequences in batch in the descending order of length

        sorted_len, idx = len_x.sort(dim=0, descending=True)
        sorted_x = x[idx]

        # Convert to packed sequence batch
        packed_x = pack_padded_sequence(sorted_x, lengths=sorted_len, batch_first=True)

        # Check init_state
        if initial_state is not None:
            if isinstance(initial_state, tuple):  # (h_0, c_0) in LSTM
                hx = [state[:, idx] for state in initial_state]
            else:
                hx = initial_state[:, idx]  # h_0 in RNN
        else:
            hx = None

        # Do forward pass
        self.rnn_module.flatten_parameters()
        packed_output, last_s = self.rnn_module(packed_x, hx)

        # pad the packed_output
        max_seq_len = x.size(1)
        padded_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_seq_len
        )

        # Reverse to the original order
        _, reverse_idx = idx.sort(dim=0, descending=False)
        padded_output = padded_output[reverse_idx]

        return padded_output, last_s


def collate_fn_pad_seq(data):
    """Pad sequences function used as collate_fn in DataLoader. Return as dict."""
    feat, labels, lengths = zip(*data)
    feat_padded = pad_sequence(feat, batch_first=True)
    return BatchDict(
        {
            "subseq": feat_padded,
            "nametype": feat_padded,
            "data": torch.stack([feat_padded, feat_padded * 2, feat_padded * 4]),
            "control": torch.stack([feat_padded, feat_padded * 1.5])[:, :, :-10, :],
            "labels": torch.Tensor(labels).long(),
            "subseq_lens": torch.Tensor(lengths).long(),
        }
    )
