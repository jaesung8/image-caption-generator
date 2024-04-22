import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from multiprocessing import Manager
import numpy as np

from src.constants import DATA_DIR_PATH


# class FlickrDataset(Dataset):
#     def __init__(self, images, captions, targets, start=0, end=1):
#         image_num = len(images)
#         start_index = int(image_num * start)
#         end_index = int(image_num * end)
#         self.images = images[start_index:end_index]
#         self.captions = captions[start_index*5:end_index*5]
#         self.targets = targets[start_index*5:end_index*5]

#     def __len__(self):
#         return len(self.captions)

#     def __getitem__(self, index):
#         # input_caption = torch.cat(
#         #     torch.tensor(self.vocab['<SOS>']), self.captions[index]
#         # )
#         # output_caption = torch.cat(
#         #     self.captions[index], torch.tensor(self.vocab['<EOS>'])
#         # )
#         return self.images[index//5], self.captions[index], self.targets[index]


class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, img, encoded_cpation):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (img, encoded_cpation)


class FlickrDataset(Dataset):
    def __init__(self, caption_lines, vocab, transform=None, train=True):
        self.transform = transform
        self.caption_lines = caption_lines
        self.vocab = vocab
        self.train = train

        # self.cache = DatasetCache(Manager())

    def __len__(self):
        if self.train:
            return len(self.caption_lines)
        else:
            return len(self.caption_lines) // 5

    def __getitem__(self, index):
        # if self.cache.is_cached(index):
        #     return self.cache.get(index)
        
        # image_path, caption = self.caption_lines[index].split(',', 1)
        # img = Image.open(os.path.join(DATA_DIR_PATH, image_path))
        if not self.train:
            index = index * 5
        image_path, encoded_cpation, all_captions = self.caption_lines[index]
        img = Image.open(image_path)

        if self.transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            img = transform(img)
        else:
            img = self.transform(img)

        # encoded_cpation = [self.vocab[token] for token in caption.strip().replace('"', '').split()]
        # nlp = English()
        # tokenizer = Tokenizer(nlp.vocab)
        # encoded_cpation = [self.vocab[token.text] for token in tokenizer(caption.strip())]
        # encoded_cpation = [self.vocab['<SOS>']] + encoded_cpation + [self.vocab["<EOS>"]]
        img = torch.FloatTensor(np.array(img))
        encoded_cpation = torch.LongTensor(encoded_cpation)
        all_captions = torch.LongTensor(all_captions)

        # self.cache.cache(index, img, encoded_cpation)
        return img, encoded_cpation, all_captions


class MyCollate:
    def __init__(self, pad_value):
        self.pad_value = pad_value
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        img = torch.cat(imgs, dim=0)
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_value)
        return img, captions
