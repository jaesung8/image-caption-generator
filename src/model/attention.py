import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feature_size, decoder_hidden_size, hidden_size):
        super(Attention, self).__init__()
        self.u = nn.Linear(feature_size, hidden_size)
        self.w = nn.Linear(decoder_hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        self.activation = nn.Tanh()
        # self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, decoder_hidden):
        encoder_att = self.u(feature)
        decoder_att = self.w(decoder_hidden).unsqueeze(1)
        output = self.activation(encoder_att + decoder_att)
        e = self.v(output).squeeze(2)
        alpha = self.softmax(e)
        context_vector = (feature * alpha.unsqueeze(2)).sum(dim=1)
        return context_vector, alpha
