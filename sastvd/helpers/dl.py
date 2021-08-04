import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import Dataset


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
    """Pad sequences function used as collate_fn in DataLoader."""
    feat, labels, lengths = zip(*data)
    feat_padded = pad_sequence(feat, batch_first=True)
    return (
        feat_padded,
        torch.Tensor(labels).long(),
        torch.Tensor(lengths).long(),
    )
