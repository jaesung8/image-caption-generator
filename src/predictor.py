import os
import copy

import torch

from src.constants import MODEL_DIR_PATH, DATA_DIR_PATH
from src.data.image import preprocess_images
from src.data.vocab import create_vocab


class Predictor:
    def __init__(self):
        self.model = torch.load(os.path.join(MODEL_DIR_PATH, 'model.pt'))
        self.vocab = create_vocab()

        if torch.cuda.is_available():
            self.device = self.model.device
        else:
            self.device = torch.device('cpu')
            self.model.device = self.device

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_paths):
        images = preprocess_images(image_paths)
        captions = []
        for image in images:
            captions.append(self.model.sample(image, self.vocab))

        return captions
