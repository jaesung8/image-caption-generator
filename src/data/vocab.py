import os
from collections import Counter, OrderedDict

import torch
from torchtext.vocab import vocab
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from src.constants import DATA_DIR_PATH

captions = [
    'flickr8k/captions.txt',
    'flickr30k/captions.txt',
]

def create_spacy_token_vocab():
    vocab_path = os.path.join(DATA_DIR_PATH, "spacy_vocab.pt")
    if os.path.isfile(vocab_path):
        return torch.load(vocab_path)

    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    counter = Counter()

    for caption_path in captions:
        with open(f'{DATA_DIR_PATH}/{caption_path}', 'r') as caption_reader:
            image_captions = caption_reader.readlines()

    for image_caption in image_captions[1:]:
        _, caption = image_caption.split(',', 1)
        tokens = [token.text for token in tokenizer(caption.strip())]
        counter.update(tokens)

    ordered_dict = OrderedDict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))
    result = vocab(ordered_dict, specials=['<SOS>', '<EOS>', '<PAD>'])
    torch.save(result, vocab_path)
    return result


def create_word_vocab():
    vocab_path = os.path.join(DATA_DIR_PATH, "word_vocab.pt")
    if os.path.isfile(vocab_path):
        return torch.load(vocab_path)
    words = []
    for caption_path in captions:
        with open(f'{DATA_DIR_PATH}/{caption_path}', 'r') as caption_reader:
            image_captions = caption_reader.readlines()

        for image_caption in image_captions[1:]:
            _, caption = image_caption.split(',', 1)
            words.extend(caption.strip().replace('"', '').split())

    words.extend(['<SOS>', '<EOS>', '<PAD>'])
    words = list(set(words))
    vocab = {val:index for index, val in enumerate(words)}
    torch.save(vocab, vocab_path)
    return vocab
