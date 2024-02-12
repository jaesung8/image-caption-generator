import torch
import torch.nn as nn

from .attention import Attention


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embedding(captions))
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        _, h_c = self.lstm(features.unsqueeze(1))
        outputs, _ = self.lstm(embeddings, h_c)
        outputs = self.linear(outputs)
        return outputs


class AttentionDecoder(nn.Module):
    def __init__(self, device, embedding_size, hidden_size, attention_size, vocab_size, feature_size=2048):
        super(AttentionDecoder, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm_cell = nn.LSTMCell(embedding_size + feature_size, hidden_size)
        self.init_h = nn.Linear(feature_size, hidden_size)
        self.init_c = nn.Linear(feature_size, hidden_size)
        self.attention = Attention(feature_size, hidden_size, attention_size)

        self.f_beta = nn.Linear(hidden_size, feature_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, features, captions):
        batch_size = features.size(0)
        vocab_size = self.vocab_size
        length = captions.size(1)

        # Flatten image
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        # num_pixels = encoder_out.size(1)

        # Embedding
        embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)
        
        feature_mean = features.mean(dim=1)
        c = self.init_c(feature_mean)
        h = self.init_h(feature_mean)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        # decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, length, vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, length, features.size(1)).to(self.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(length):
            # batch_size_t = sum([l > t for l in decode_lengths])
            context, alpha = self.attention(features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context
            h, c = self.lstm_cell(torch.cat([embeddings[:, t], gated_context], dim=1), (h, c))
            preds = self.fc(self.dropout(h))
            predictions[:, t] = preds
            alphas[:, t] = alpha

        return predictions, alphas