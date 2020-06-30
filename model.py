import torch
from torch import nn
from torch import Tensor


class Model(nn.Module):

    def __init__(self, vocab_size: int,
                 hidden_size: int = 300,
                 dropout: float = 0.1):

        super(Model, self).__init__()

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=2,
                            dropout=dropout,
                            batch_first=True)
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, input: Tensor):

        embeddings = self.embed(input)
        lstm_out, _ = self.lstm(embeddings)
        output = self.hidden2vocab(lstm_out)
        return output

