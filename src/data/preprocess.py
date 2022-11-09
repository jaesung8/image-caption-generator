import glob
import os

from PIL import Image
import torch
from torchvision import transforms
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from src.constants import DATA_DIR_PATH


def preprocess_flickr_8k(vocab):
    image_tensor_path = os.path.join(DATA_DIR_PATH, 'image_tensor.pt')
    caption_tensor_path = os.path.join(DATA_DIR_PATH, 'caption_tensor.pt')
    if os.path.isfile(image_tensor_path) and os.path.isfile(caption_tensor_path):
        return torch.load(image_tensor_path), torch.load(caption_tensor_path)

    image_dir_path = f'{DATA_DIR_PATH}/images'
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tokenizer = Tokenizer(vocab)

    with open(f'{DATA_DIR_PATH}/captions.txt', 'r') as caption_reader:
        image_captions = caption_reader.readlines()

    prev_image_name = None
    processed_image_tensor = torch.empty(size=(len(image_captions)-1, 1000))
    processed_caption_tensor = torch.empty(size=(len(image_captions)-1, 50))
    for i, line in enumerate(image_captions[1:]):
        image_name, caption = line.strip().split(',', 1)
        processed_caption_tensor[i] = torch.tensor(
            [vocab[token] for token in tokenizer(caption.strip())] + [vocab['<EOS>']]
        )
        if image_name == prev_image_name:
            processed_image_tensor[i] = prev_image_tensor
        else:
            prev_image_name = image_name
            image = Image.open(os.path.join(image_dir_path, image_name))
            prev_image_tensor = preprocess(image)
            processed_image_tensor[i] = prev_image_tensor

    processed_image_tensor.save(f'{DATA_DIR_PATH}/image_tensor.pt')
    processed_caption_tensor.save(f'{DATA_DIR_PATH}/caption_tensor.pt')
    return processed_image_tensor, processed_caption_tensor
