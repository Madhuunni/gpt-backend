import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """Same embedding + bidirectional GRU architecture as the earlier project."""
    def __init__(self, vocabulary_size, embedding_size=32, hidden_size=48):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.encoder = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)

    def encode(self, tokens, lengths):
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, hidden = self.encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=tokens.shape[1])
        sentence = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return encoded, sentence
