import torch
import math
from torch import nn
from torch import Tensor


class Model(nn.Module):

    def __init__(self, device,
                 vocab_size: int,
                 hidden_size: int = 300,
                 n_heads: int = 5,
                 dropout: float = 0.1):

        super(Model, self).__init__()
        self.device = device
        self.embed = nn.Embedding(vocab_size, hidden_size)
        #self.pe = PositionalEncoding(hidden_size)
        #self.self_attention = MultiHeadedAttention(num_heads=n_heads, size=hidden_size, dropout=dropout)
        #self.ReLU = nn.ReLU()
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=4,
                            dropout=dropout,
                            batch_first=True)
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, input: Tensor):

        input = input.to(self.device)
        embeddings = self.embed(input)#self.pe(self.embed(input))
        #attention_out = self.self_attention(embeddings, embeddings, embeddings)
        #attention_out = self.ReLU(attention_out)
        lstm_out, _ = self.lstm(embeddings)#self.lstm(attention_out)
        output = self.hidden2vocab(lstm_out)
        return output

#i copied this from JoeyNMT
# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size)

        output = self.output_layer(context)

        return output

class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.
    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """
    def __init__(self,
                 size: int = 0,
                 max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 2, dtype=torch.float) *
                              -(math.log(10000.0) / size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, :emb.size(1)]
