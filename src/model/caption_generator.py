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
        vocab_size,
        num_layers,
    ):
        super(CaptionGenerator, self).__init__()
        self.device = device
        self.encoder = Encoder(embedding_size).to(self.device)
        self.decoder = Decoder(
            embedding_size, hidden_size, vocab_size, num_layers
        ).to(self.device)
        self.max_length = 50

    def forward(self, images, captions):
        features = self.encoder(images)
        output = self.decoder(features, captions)
        return output

    def greedy_generate(self, image, vocab):
        caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(self.max_length):
                hiddens, states = self.lstm(x, states)
                output = self.fc(hiddens)
                predicted = output.argmax(1)
                x = self.decoder.embedding(predicted).unsqueeze(0)
                token = vocab.lookup_token[predicted.item()]
                if token == "<EOS>":
                    break
                caption.append(token)
        return caption

    def sample(self, image, vocab):
        caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(self.max_length):
                hiddens, states = self.lstm(x, states)
                output = self.fc(hiddens)
                dist = Categorical(F.softmax(output, dim=-1))
                predicted = dist.sample()
                x = self.decoder.embedding(predicted).unsqueeze(0)
                token = vocab.lookup_token[predicted.item()]
                if token == "<EOS>":
                    break
                caption.append(token)
        return caption

    def sample_beam_search(self, features, vocab, k=4):
        vocab_size = len(vocab)
        encoder_size = features.size(-1)
        features = features.view(1, 1, encoder_size)
        inputs = features.expand(k, 1, encoder_size)

        top_k_scores = torch.zeros(k, 1).to(self.device)
        seqs = torch.zeros(k, 1).long().to(self.device)
        complete_seqs = list()
        complete_seqs_scores = list()
        
        step = 1
        hidden, cell = None, None
        
        while True:
            if step == 1:
                outputs, (hidden, cell) = self.lstm(inputs, None)
            else:
                outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))

            outputs = self.linear(outputs.squeeze(1))
            scores = F.log_softmax(outputs, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            
            # 첫번째 스텝은 모두 같은 score을 가진다 <SOS>라는 뿌리로 시작하기 때문
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)  # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)  # (s)

            # Score를 정립시켜서 index를 구한다.
            # prev_word_inds : tensor([0, 0, 1, 0], device='cuda:0') 
            # next_word_inds : tensor([78, 30, 50, 31], device='cuda:0')
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # 새로둔 단어를 seqs에 더한다.
            if step==1:
                seqs = next_word_inds.unsqueeze(1)
            else:
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <EOS>, <EOS>idx == 2)?
            # 인덱스 마지막 저장하기
            incomplete_inds = [
                ind for ind, next_word in enumerate(next_word_inds)
                if next_word != vocab('<EOS>')
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # 마지막 단어 세팅
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            hidden = hidden[:, prev_word_inds[incomplete_inds]]
            cell = cell[:, prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            inputs = self.embed(k_prev_words)
            if step > self.max_seg_length:
                break
            step += 1