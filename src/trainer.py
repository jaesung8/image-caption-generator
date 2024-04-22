import os
import time
import sys
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu

from src.model.caption_generator import CaptionGenerator
from src.model.attention_caption_generator import AttentionCaptionGenerator
from src.data.vocab import create_spacy_token_vocab, create_word_vocab
from src.data.preprocess import preprocess_flickr, preprocess_caption
from src.data.dataset import FlickrDataset, MyCollate
from src.constants import MODEL_DIR_PATH, DATA_DIR_PATH
from src.utils import AverageMeter, accuracy, calculate_caption_lengths


class Trainer:
    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        self._init_hyperparameters(hyperparameters)

        if torch.cuda.is_available(): 
            self.device = torch.device(self.device_number)
        else:
            self.device = torch.device('cpu')        
        # self.vocab = create_spacy_token_vocab()
        self.vocab = create_word_vocab()
        # images, input, output = preprocess_flickr(self.vocab)
        # self.train_data_loader = DataLoader(
        #     FlickrDataset(images, input, self.vocab, end=0.8),
        #     batch_size=96,
        #     num_workers=2,
        # )
        # self.valid_data_loader = DataLoader(
        #     FlickrDataset(images, input, output, start=0.8, end=0.9),
        #     batch_size=96,
        # )
        train_captions, valid_captions, _ = preprocess_caption()
        batch_size = 96

        self.train_data_loader = DataLoader(
            FlickrDataset(train_captions, self.vocab),
            batch_size=batch_size,
            num_workers=8,
            collate_fn=MyCollate(self.vocab["<PAD>"]),
        )
        self.valid_data_loader = DataLoader(
            FlickrDataset(valid_captions, self.vocab),
            batch_size=batch_size,
            num_workers=2,
            collate_fn=MyCollate(self.vocab["<PAD>"]),
        )
        self.train_data_num = len(self.train_data_loader)
        self.valid_data_num = len(self.valid_data_loader)
        self.model = CaptionGenerator(
            self.device,
            embedding_size=256,
            hidden_size=256,
            vocab=self.vocab,
            num_layers=2,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab["<PAD>"])

        self.log_interval = int(self.train_data_num / 16)


    def _init_hyperparameters(self, hyperparameters):
        for k, v in hyperparameters.items():
            setattr(self, k, v)

    def train(self):
        min_valid_loss, cur_valid_loss = sys.maxsize, 0
        train_loss_list, valid_loss_list = [], []
        train_loss, train_top1, train_top5 = AverageMeter(), AverageMeter(), AverageMeter()
        valid_loss, valid_top1, valid_top5 = AverageMeter(), AverageMeter(), AverageMeter()

        for epoch in range(self.num_epochs):
            start_time = time.time()
            cur_valid_loss = 0
            cur_valid_caption_length = 0
            for i, (images, captions) in enumerate(self.train_data_loader):
                batch_start_time = time.time()
                images = images.to(self.device)
                captions = captions.to(self.device)
                targets = captions[:, 1:]
                outputs = self.model(images, captions)
                outputs = torch.permute(outputs, (0, 2, 1))
                outputs = outputs[:, :, :-1]
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                total_caption_length = calculate_caption_lengths(self.vocab, captions)
                acc1 = accuracy(outputs, targets, 1, self.vocab["<PAD>"]) / total_caption_length
                acc5 = accuracy(outputs, targets, 5, self.vocab["<PAD>"]) / total_caption_length
                train_loss.update(loss.item(), total_caption_length)
                train_top1.update(acc1, total_caption_length)
                train_top5.update(acc5, total_caption_length)
                if i and i % self.log_interval == 0:
                    print(
                        f'[Train] Epoch {epoch+1}  Batch {i+1}  '
                        f'Loss: {train_loss.val:.5f} {train_loss.avg:.5f}  ' 
                        f'Top 1 Accuracy {train_top1.val:.3f} ({train_top1.avg:.3f})  '
                        f'Top 5 Accuracy {train_top5.val:.3f} ({train_top5.avg:.3f})  '
                        f'Time: {time.time() - batch_start_time}s'
                    )
            self.scheduler.step()

            with torch.no_grad():
                for i, (images, captions) in enumerate(self.valid_data_loader):
                    batch_start_time = time.time()
                    images = images.to(self.device)
                    captions = captions.to(self.device)
                    targets = captions[:, 1:]
                    outputs = self.model(images, captions)
                    outputs = torch.permute(outputs, (0, 2, 1))
                    outputs = outputs[:, :, :-1]
                    loss = self.criterion(outputs, targets)
                    cur_valid_loss += loss.item()

                    total_caption_length = calculate_caption_lengths(self.vocab, captions)
                    cur_valid_caption_length += total_caption_length
                    acc1 = accuracy(outputs, targets, 1, self.vocab["<PAD>"]) / total_caption_length
                    acc5 = accuracy(outputs, targets, 5, self.vocab["<PAD>"]) / total_caption_length
                    valid_loss.update(loss.item(), total_caption_length)
                    valid_top1.update(acc1, total_caption_length)
                    valid_top5.update(acc5, total_caption_length)
                    if i and i % self.log_interval == 0:
                        print(
                            f'[Valid] Epoch {epoch+1}  Batch {i+1}  '
                            f'Loss: {valid_loss.val:.5f} {valid_loss.avg:.5f}  ' 
                            f'Top 1 Accuracy {valid_top1.val:.3f} ({valid_top1.avg:.3f})  '
                            f'Top 5 Accuracy {valid_top5.val:.3f} ({valid_top5.avg:.3f})  '
                            f'Time: {time.time() - batch_start_time}s'
                        )

            if min_valid_loss > cur_valid_loss / cur_valid_caption_length:
                torch.save(self.model.state_dict(), os.path.join(MODEL_DIR_PATH, 'model.pt'))
                min_valid_loss = cur_valid_loss / cur_valid_caption_length

            print(f'Epoch {epoch + 1} finished {time.time() - start_time}s elapsed')
        plt.plot(train_loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        # plt.savefig('train_loss.png')
        plt.plot(valid_loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig('loss.png')


class AttentionTrainer:
    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        self._init_hyperparameters(hyperparameters)

        if torch.cuda.is_available(): 
            self.device = torch.device(self.device_number)
        else:
            self.device = torch.device('cpu')   
        # self.vocab = create_spacy_token_vocab()
        self.vocab = create_word_vocab()
        # images, input, output = preprocess_flickr(self.vocab)
        # self.train_data_loader = DataLoader(
        #     FlickrDataset(images, input, self.vocab, end=0.8),
        #     batch_size=96,
        #     num_workers=2,
        # )
        # self.valid_data_loader = DataLoader(
        #     FlickrDataset(images, input, output, start=0.8, end=0.9),
        #     batch_size=96,
        # )
        train_captions, valid_captions, _ = preprocess_caption(self.vocab)
        batch_size = 200

        self.train_data_loader = DataLoader(
            FlickrDataset(train_captions, self.vocab),
            batch_size=batch_size,
            num_workers=8,
            collate_fn=MyCollate(self.vocab["<PAD>"]),
            # pin_memory=True,
        )
        self.valid_data_loader = DataLoader(
            FlickrDataset(valid_captions, self.vocab),
            batch_size=batch_size,
            num_workers=2,
            collate_fn=MyCollate(self.vocab["<PAD>"]),
            # pin_memory=True,
        )
        self.train_data_num = len(self.train_data_loader)
        self.valid_data_num = len(self.valid_data_loader)
        self.model = AttentionCaptionGenerator(
            self.device,
            embedding_size=512,
            hidden_size=512,
            attention_size=512,
            vocab=self.vocab,
        ).to(self.device)
        # self.model = nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab["<PAD>"])

        self.log_interval = int(self.train_data_num / 8)


    def _init_hyperparameters(self, hyperparameters):
        for k, v in hyperparameters.items():
            setattr(self, k, v)

    def train(self):
        min_valid_loss, cur_valid_loss = sys.maxsize, 0
        train_loss_list, train_top1_acc, train_top5_acc = [], [], []
        valid_loss_list, valid_top1_acc, valid_top5_acc = [], [], []
        

        for epoch in range(self.num_epochs):
            train_loss, train_top1, train_top5 = AverageMeter(), AverageMeter(), AverageMeter()
            valid_loss, valid_top1, valid_top5 = AverageMeter(), AverageMeter(), AverageMeter()
            start_time = time.time()
            cur_valid_loss = 0
            cur_valid_caption_length = 0
            for i, (images, captions) in enumerate(self.train_data_loader):
                batch_start_time = time.time()
                images = images.to(self.device)
                captions = captions.to(self.device)
                targets = captions[:, 1:]
                outputs, alphas = self.model(images, captions)
                outputs = torch.permute(outputs, (0, 2, 1))
                outputs = outputs[:, :, :-1]
                loss = self.criterion(outputs, targets)
                loss += self.alpha_lambda * ((1 - alphas.sum(1))**2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                total_caption_length = calculate_caption_lengths(self.vocab, captions)
                acc1 = accuracy(outputs, targets, 1, self.vocab["<PAD>"]) / total_caption_length
                acc5 = accuracy(outputs, targets, 5, self.vocab["<PAD>"]) / total_caption_length
                train_loss.update(loss.item(), total_caption_length)
                train_top1.update(acc1, total_caption_length)
                train_top5.update(acc5, total_caption_length)
                if i and i % self.log_interval == 0:
                    print(
                        f'[Train] Epoch {epoch+1}  Batch {i+1}  '
                        f'Loss: {train_loss.val:.5f} {train_loss.avg:.5f}  ' 
                        f'Top 1 Accuracy {train_top1.val:.3f} ({train_top1.avg:.3f})  '
                        f'Top 5 Accuracy {train_top5.val:.3f} ({train_top5.avg:.3f})  '
                        f'Time: {time.time() - batch_start_time}s'
                    )
                    train_loss_list.append(train_loss.val)
                    train_top1_acc.append(train_top1.val)
                    train_top5_acc.append(train_top5.val)
            self.scheduler.step()

            with torch.no_grad():
                for i, (images, captions) in enumerate(self.valid_data_loader):
                    batch_start_time = time.time()
                    images = images.to(self.device)
                    captions = captions.to(self.device)
                    targets = captions[:, 1:]
                    outputs, alphas = self.model(images, captions)
                    outputs = torch.permute(outputs, (0, 2, 1))
                    outputs = outputs[:, :, :-1]
                    loss = self.criterion(outputs, targets)
                    cur_valid_loss += loss.item()

                    total_caption_length = calculate_caption_lengths(self.vocab, captions)
                    cur_valid_caption_length += total_caption_length
                    acc1 = accuracy(outputs, targets, 1, self.vocab["<PAD>"]) / total_caption_length
                    acc5 = accuracy(outputs, targets, 5, self.vocab["<PAD>"]) / total_caption_length
                    valid_loss.update(loss.item(), total_caption_length)
                    valid_top1.update(acc1, total_caption_length)
                    valid_top5.update(acc5, total_caption_length)
                    
                    if i and i % self.log_interval == 0:
                        print(
                            f'[Valid] Epoch {epoch+1}  Batch {i+1}  '
                            f'Loss: {valid_loss.val:.5f} {valid_loss.avg:.5f}  ' 
                            f'Top 1 Accuracy {valid_top1.val:.3f} ({valid_top1.avg:.3f})  '
                            f'Top 5 Accuracy {valid_top5.val:.3f} ({valid_top5.avg:.3f})  '
                            f'Time: {time.time() - batch_start_time}s'
                        )
                        valid_loss_list.append(valid_loss.val)
                        valid_top1_acc.append(valid_top1.val)
                        valid_top5_acc.append(valid_top5.val)

            if min_valid_loss > cur_valid_loss / cur_valid_caption_length:
                torch.save(self.model.state_dict(), os.path.join(MODEL_DIR_PATH, 'attention_model.pt'))
                min_valid_loss = cur_valid_loss / cur_valid_caption_length

            print(f'Epoch {epoch + 1} finished {time.time() - start_time}s elapsed')

            train_data = {
                "train_loss_list": train_loss_list,
                "train_top1_acc": train_top1_acc,
                "train_top5_acc": train_top5_acc,
                "valid_loss_list": valid_loss_list,
                "valid_top1_acc": valid_top1_acc,
                "valid_top5_acc": valid_top5_acc,
            }

            with open('train_data.pkl', 'wb') as caption_writer:
                pkl.dump(train_data, caption_writer)
        # plt.plot(train_loss_list)
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # # plt.savefig('train_loss.png')
        # plt.plot(valid_loss_list)
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.savefig('attention_loss.png')
