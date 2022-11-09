import os
from collections import Counter, OrderedDict

import torch
from torchtext.vocab import vocab
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from src.constants import DATA_DIR_PATH


def create_vocab():
    vocab_path = os.path.join(DATA_DIR_PATH, "vocab.pth")
    if os.path.isfile(vocab_path):
        return torch.load(vocab_path)

    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    counter = Counter()

    with open(f'{DATA_DIR_PATH}/captions.txt', 'r') as caption_reader:
        image_captions = caption_reader.readlines()

    for image_caption in image_captions[1:]:
        _, caption = image_caption.split(',', 1)
        tokens = tokenizer(caption.strip())
        counter.update(tokens)

    ordered_dict = OrderedDict(
        sorted(counter.items(), key=lambda x: x[1], reverse=True),
        specials=["<EOS>", "<PAD>"],
    )
    result = vocab(ordered_dict)
    result.save(vocab_path)
    return result