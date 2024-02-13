import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
import os

from src.constants import DATA_DIR_PATH


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_att(image_path, alphas, words, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    print(image_path, words)
    image = Image.open(image_path)
    base_image_path = os.path.basename(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    alphas = torch.tensor(alphas)
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.reshape(7, 7), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.reshape(7, 7), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()
    plt.savefig(f'{base_image_path}.png')
    plt.clf()


def accuracy(preds, targets, k, ignore_index):
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    targets = targets.unsqueeze(1).expand_as(pred)
    correct = pred.eq(targets)
    correct = torch.where(correct == ignore_index, 0, correct)
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * 100.0


def calculate_caption_lengths(vocab, captions):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (vocab['<SOS>'], vocab['<EOS>'], vocab['<PAD>']):
                continue
            else:
                lengths += 1
    return lengths
