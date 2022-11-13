import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
        images, input, output = preprocess_flickr_8k(self.vocab)
        self.train_data_loader = DataLoader(
            ImageCatpionDataset(images, input, output, end=0.8),
            batch_size=128,
        )
        self.valid_data_loader = DataLoader(
            ImageCatpionDataset(images, input, output, start=0.8, end=0.9),
            batch_size=128,
        )
        self.train_data_num = len(self.train_data_loader)
        self.valid_data_num = len(self.valid_data_loader)
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

    def _init_hyperparameters(self, hyperparameters):
        for k, v in hyperparameters.items():
            setattr(self, k, v)

    def train(self):
        min_valid_loss = None
        train_loss_list = []
        for epoch in range(self.num_epochs):
            train_loss, valid_loss = 0.0, 0.0
            for i, (images, captions, targets) in enumerate(self.train_data_loader):
                images = images.to(self.device)
                captions = captions.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images, captions)
                outputs = torch.permute(outputs, (0, 2, 1))
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
            self.scheduler.step()

            with torch.no_grad():
                for i, (images, captions, targets) in enumerate(self.valid_data_loader):
                    images = images.to(self.device)
                    captions = captions.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(images, captions)
                    outputs = torch.permute(outputs, (0, 2, 1))
                    loss = self.criterion(outputs, targets)
                    valid_loss += loss.item()

            train_loss = train_loss / self.train_data_num
            train_loss_list.append(train_loss)
            print(
                f'Epoch {epoch+1}  '
                f'Training Loss: {train_loss:.6f}  ' 
                f'Validation Loss: {valid_loss / self.valid_data_num:.6f}'
            )

            if not min_valid_loss or min_valid_loss > loss:
                torch.save(self.model, os.path.join(MODEL_DIR_PATH, 'model.pt'))
                min_valid_loss = loss

        plt.plot(train_loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig('train_loss.png')
        plt.savefig('train_loss.png')
