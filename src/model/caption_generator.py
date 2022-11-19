import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from src.model.encoder import Encoder
from src.model.decoder import Decoder


class CaptionGenerator(nn.Module):
    def __init__(
        self,
        device,
        embedding_size,
        hidden_size,
        vocab,
        num_layers,
    ):
        super(CaptionGenerator, self).__init__()
        self.device = device
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.encoder = Encoder(embedding_size).to(self.device)
        self.decoder = Decoder(
            embedding_size, hidden_size, self.vocab_size, num_layers
        ).to(self.device)
        self.max_length = 50

    def forward(self, images, captions):
        features = self.encoder(images)
        output = self.decoder(features, captions)
        return output

    def greedy_generate(self, image):
        caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            _, states = self.decoder.lstm(x)
            x = self.decoder.embedding(self.vocab["<SOS>"]).unsqueeze(0)

            for _ in range(self.max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.fc(hiddens)
                predicted = output.argmax(1)
                x = self.decoder.embedding(predicted).unsqueeze(0)
                token = self.vocab.lookup_token[predicted.item()]
                if token == "<EOS>":
                    break
                caption.append(token)
        return caption

    def sample(self, image):
        caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            _, states = self.decoder.lstm(x)
            x = self.decoder.embedding(self.vocab["<SOS>"]).unsqueeze(0)

            for _ in range(self.max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.fc(hiddens)
                dist = Categorical(F.softmax(output, dim=-1))
                predicted = dist.sample()
                x = self.decoder.embedding(predicted).unsqueeze(0)
                token = self.vocab.lookup_token[predicted.item()]
                if token == "<EOS>":
                    break
                caption.append(token)
        return caption

    def beam_search(self, image, k=5):
        with torch.no_grad():
            features = self.encoder(image).unsqueeze(0)

            top_k_scores = torch.zeros(k, 1).to(self.device)
            seqs = torch.zeros(k, 1).long().to(self.device)
            complete_seqs = list()
            complete_seqs_scores = list()

            step = 1
            _, (hidden, cell) = self.lstm(features)
            inputs = self.decoder.embedding(self.vocab["<SOS>"]).unsqueeze(0)

            while True:
                if step == 1:
                    outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))
                    hiddens = hidden.tile((k,))
                    cells = cell.tile((k,))
                else:
                    outputs, (hiddens, cells) = self.lstm(inputs, (hiddens, cells))
                outputs = self.linear(outputs.squeeze(1))
                scores = F.log_softmax(outputs, dim=1)

                if step != 1:
                    scores = top_k_scores.expand_as(scores) + scores

                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)

                prev_word_inds = top_k_words // self.vocab_size
                next_word_inds = top_k_words % self.vocab_size
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

                incomplete_inds = [
                    ind for ind, next_word in enumerate(next_word_inds)
                    if next_word != self.vocab['<EOS>']
                ]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # 마지막 단어 세팅
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                    k -= len(complete_inds)

                if k == 0:
                    break

                seqs = seqs[incomplete_inds]
                hidden = hidden[:, prev_word_inds[incomplete_inds]]
                cell = cell[:, prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
                inputs = self.decoder.embedding(k_prev_words)
                if step > self.max_length:
                    break
                step += 1

        captions = []
        for complete_seq in complete_seqs:
            captions.append([
                self.vocab.lookup_token[i] for i in complete_seq
            ])

        return captions, complete_seqs_scores
