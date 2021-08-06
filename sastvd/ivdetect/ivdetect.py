import torch
import torch.nn as nn


class GRU(nn.Module):
    """Basic GRU."""

    def __init__(self, input_size, hidden_size, num_layers):
        """Initilisation."""
        super(GRU, self).__init__()
        self.dropout = nn.Dropout(p=0.4)
        self.gru = dl.DynamicRNN(
            nn.GRU(input_size, hidden_size, num_layers, dropout=0.5)
        )

    def forward(self, x: torch.Tensor, x_len: list):
        """Forward pass.

        Args:
            x (torch.Tensor): (BATCH_SIZE, MAX_PADDED_LEN, EMBEDDING_SIZE)
            x_len (list): Lengths of each of the sentences in the batch
        """
        out, hidden = self.gru(x, x_len)
        out = out[range(out.shape[0]), x_len - 1, :]
        out = self.dropout(out)
        return out, hidden


class IVDetect(nn.Module):
    """IVDetect Model."""

    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        """Initialise model."""
        super().__init__()
        self.gru1 = GRU(input_size, hidden_size, num_layers)
        self.gru2 = GRU(input_size, hidden_size, num_layers)
        self.gru3 = GRU(input_size, hidden_size, num_layers)
        self.gru4 = GRU(input_size, hidden_size, num_layers)

        self.bigru = nn.GRU(input_size, hidden_size, num_layers, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: dict):
        """Forward pass.

        Args:
            x (dict): X is a dict containing the relevant information. See example.

        Dict arguments:
        feat - Feature with shape: (BATCH_SIZE, MAX_PADDED_LEN, EMBEDDING_SIZE)
        lens - Lengths of each of the sentences in the batch.

        Attention
        self.att = nn.MultiheadAttention(input_size, 10, batch_first=True)
        out = self.att(out, out, out)[0]
        """
        out_gru1 = self.gru1(x.subseq, x.subseq_len)

        out_gru2 = self.gru1(x.nametype, x.nametype_len)

        dd_lines = []
        for dd_line in x.data:
            dd_lines.append(self.gru3(dd_line))
        print(dd_lines[0])

        stacked = torch.stack([out_gru1[0], out_gru2[0]], dim=1)

        out_bigru = self.bigru(stacked)
        out_fc = self.fc(out_bigru[0][:, -1, :])

        # padded = pad_sequence([out_gru1[0], out_gru2[0]], batch_first=True)
        # print(padded.shape)

        return out_fc, 1
