import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embedding(captions))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        outpus, _ = self.lstm(embeddings)
        outputs = self.linear(outpus)
        return outputs
