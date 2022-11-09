import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.caption_generator import CaptionGenerator
from src.data.vocab import create_vocab
from src.data.preprocess import preprocess_flickr_8k
from src.data.dataset import ImageCatpionDataset
from src.constants import MODEL_DIR_PATH


class Trainer:
    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        if torch.cuda.is_available(): 
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self._init_hyperparameters(hyperparameters)
        self.vocab = create_vocab()
        images, captions = preprocess_flickr_8k(self.vocab)
        self.data_loader = DataLoader(
            ImageCatpionDataset(images, captions),
            batch_size=128,
        )
        self.model = CaptionGenerator(
            self.device,
            embedding_size=256,
            hidden_size=256,
            vocab_size=len(self.vocab),
            num_layers=2,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab["<PAD>"])


    def train(self):
        prev_loss = None
        for epoch in range(self.num_epochs):
            for i, (images, captions) in enumerate(self.data_loader):
                images.to(self.device)
                captions.to(self.device)
                outputs = self.model(images, captions[:-1])
                loss = self.criterion(outputs, captions)
                self.optimizer.zero_grad()
                loss.bacward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            self.scheduler.step()

            if epoch and epoch % self.epoch_interval == 0:
                print(epoch, loss)
                if not prev_loss or prev_loss > loss:
                    self.model.save(os.path.join(MODEL_DIR_PATH, 'model.pt'))
                    prev_loss = loss
