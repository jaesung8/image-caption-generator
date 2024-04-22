import os
import copy

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import MODEL_DIR_PATH, DATA_DIR_PATH
from src.data.preprocess import preprocess_caption
from src.data.vocab import create_word_vocab, create_spacy_token_vocab
from src.model.caption_generator import CaptionGenerator
from src.data.dataset import FlickrDataset, MyCollate

from src.model.attention_caption_generator import AttentionCaptionGenerator


class Predictor:
    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        self._init_hyperparameters(hyperparameters)

        if torch.cuda.is_available():
            self.device = torch.device(self.device_number)
        else:
            self.device = torch.device('cpu')

        self.vocab = create_word_vocab()

        self.model = CaptionGenerator(
            self.device,
            embedding_size=256,
            hidden_size=256,
            vocab=self.vocab,
            num_layers=2
        )
        self.model.load_state_dict(torch.load(os.path.join(MODEL_DIR_PATH, 'model.pt')))
        self.model.to(self.device)
        self.model.eval()

        batch_size = 96
        _, _, test_captions = preprocess_caption()
        self.test_data_loader = DataLoader(
            FlickrDataset(test_captions, self.vocab),
            batch_size=batch_size,
            collate_fn=MyCollate(self.vocab["<PAD>"]),
        )

        self.itos_map = {v:k for k, v in self.vocab.items()}

    def _init_hyperparameters(self, hyperparameters):
        for k, v in hyperparameters.items():
            setattr(self, k, v)

    def predict(self):
        entire_captions = []
        for i, (images, captions) in enumerate(self.test_data_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            infered_captions = self.model.greedy(images)
            print(infered_captions)
            for caption in infered_captions:
                words = [self.itos_map[idx] for idx in caption]
                print(words)
            break

        return entire_captions



class AttentionPredictor:
    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        self._init_hyperparameters(hyperparameters)

        # if torch.cuda.is_available():
        #     self.device = torch.device(self.device_number)
        # else:
        self.device = torch.device('cpu')

        self.vocab = create_word_vocab()

        self.model = AttentionCaptionGenerator(
            self.device,
            embedding_size=512,
            hidden_size=512,
            attention_size=512,
            vocab=self.vocab,
        )
        self.model.load_state_dict(torch.load(os.path.join(MODEL_DIR_PATH, 'attention_model_512.pt')))
        self.model.to(self.device)
        self.model.eval()

        _, _, test_captions = preprocess_caption(self.vocab)
        self.test_dataset = FlickrDataset(test_captions, self.vocab, train=False)
        self.test_data_loader = DataLoader(
            self.test_dataset
        )

        self.itos_map = {v:k for k, v in self.vocab.items()}

    def _init_hyperparameters(self, hyperparameters):
        for k, v in hyperparameters.items():
            setattr(self, k, v)

    def predict(self):
        alphas_list = []
        words_list = []
        infered_caption_list = []
        path_cpation_list = []
        all_caption_list = []
        print(len(self.test_dataset))
        for i, (image, caption, all_captions) in tqdm(enumerate(self.test_data_loader)):
            image = image.to(self.device)
            caption = caption.to(self.device)
            infered_caption, alphas = self.model.caption(image, 5)
            if len(infered_caption):
                words = [self.itos_map[idx] for idx in infered_caption]
                alphas_list.append(alphas)
                infered_caption_list.append(infered_caption)
                words_list.append(words)
                all_caption_list.append(all_captions.squeeze(0))
                path_cpation_list.append(self.test_dataset.caption_lines[i*5][0])

        bleu_score = self.get_bleu_score(infered_caption_list, all_caption_list)
        top_idices = np.argsort(bleu_score)[-20:]
        print(np.mean(bleu_score))
        for idx in top_idices:
            _, encoded_cpation, _ = self.test_dataset[idx]
            print([self.itos_map[idx] for idx in encoded_cpation.tolist()])

        return words_list, alphas_list, path_cpation_list, top_idices
    
    def get_bleu_score(self, infered_captions, all_captions):
        all_captions = [x.tolist() for x in all_captions]

        references = []
        hypotheses = []
        for i in range(len(all_captions)):
            img_captions = list(
                map(
                    lambda c: [w for w in c if w not in {self.vocab['<SOS>'], self.vocab['<PAD>'], self.vocab['<EOS>']}],
                    all_captions[i]
                )
            )  # remove <start> and pads
            references.append(img_captions)

        for i in range(len(infered_captions)):
            img_captions = [
                w for w in infered_captions[i] if w not in {self.vocab['<SOS>'], self.vocab['<PAD>'], self.vocab['<EOS>']}
            ]
            hypotheses.append(img_captions)

        # for i in range(len(infered_captions)):
        #     img_captions = list(
        #         map(
        #             lambda c: [w for w in c if w not in {self.vocab['<SOS>'], self.vocab['<PAD>'], self.vocab['<EOS']}],
        #             all_captions[i]
        #         )
        #     )  # remove <start> and pads
        #     references.append(img_captions)


        assert len(references) == len(infered_captions)
        bleu_scores = []
        for i in range(len(all_captions)):
            bleu_scores.append(corpus_bleu([references[i]], [hypotheses[i]]))
        return bleu_scores
