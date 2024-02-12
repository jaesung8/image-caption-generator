import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from src.model.encoder import AttentionEncoder
from src.model.decoder import AttentionDecoder


class AttentionCaptionGenerator(nn.Module):
    def __init__(
        self,
        device,
        embedding_size,
        hidden_size,
        attention_size,
        vocab,
    ):
        super(AttentionCaptionGenerator, self).__init__()
        self.device = device
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.encoder = AttentionEncoder().to(self.device)
        self.decoder = AttentionDecoder(
            device, embedding_size, hidden_size, attention_size, self.vocab_size, feature_size=2048
        ).to(self.device)
        self.max_length = 50

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.encoder(images)
        outputs, alphas = self.decoder(features, captions)
        return outputs, alphas

    def greedy(self, image):
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

    def beam_search(self, image, k=5):
        with torch.no_grad():
            features = self.encoder(image).unsqueeze(0)

            top_k_scores = torch.zeros(k, 1).to(self.device)
            seqs = torch.zeros(k, 1).long().to(self.device)
            complete_seqs = list()
            complete_seqs_scores = list()

            step = 1
            _, (hidden, cell) = self.decoder.lstm(features)
            inputs = self.decoder.embedding(self.vocab["<SOS>"]).unsqueeze(0)

            while True:
                if step == 1:
                    outputs, (hidden, cell) = self.decoder.lstm(inputs, (hidden, cell))
                    hiddens = hidden.tile((k,))
                    cells = cell.tile((k,))
                else:
                    outputs, (hiddens, cells) = self.decoder.lstm(inputs, (hiddens, cells))
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

    def caption(self, image, beam_size):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        img_features = self.encoder(image).cpu()   
        img_features = img_features.expand(beam_size, -1, -1).detach()
        prev_words = torch.LongTensor([[self.vocab['<SOS>']]] * beam_size)
        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        alphas = torch.ones(beam_size, 1, img_features.size(1))

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        feature_mean = img_features.mean(dim=1)
        c = self.decoder.init_c(feature_mean)
        h = self.decoder.init_h(feature_mean)

        while True:
            embedding = self.decoder.embedding(prev_words).squeeze(1)
            context, alpha = self.decoder.attention(img_features, h)
            gate = self.decoder.sigmoid(self.decoder.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.decoder.lstm_cell(lstm_input, (h, c))
            output = self.decoder.fc(h)
            output = top_preds.expand_as(output) + output
            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True) 
            
            prev_word_idxs = top_words // output.size(1)
            next_word_idxs = top_words % output.size(1)

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != self.vocab['<EOS>']]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > self.max_length:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha