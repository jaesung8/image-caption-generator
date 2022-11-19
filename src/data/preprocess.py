import glob
import os

from PIL import Image
import torch
import torch.nn as nn 
from torchvision import transforms
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from src.constants import DATA_DIR_PATH


def preprocess_flickr_8k(vocab):
    image_tensor_path = os.path.join(DATA_DIR_PATH, 'image_tensor.pt')
    input_caption_tensor_path = os.path.join(DATA_DIR_PATH, 'input_caption_tensor.pt')
    output_caption_tensor_path = os.path.join(DATA_DIR_PATH, 'output_caption_tensor.pt')
    if (
        os.path.isfile(image_tensor_path)
        and os.path.isfile(input_caption_tensor_path)
        and os.path.isfile(output_caption_tensor_path)
    ):
        return torch.load(image_tensor_path), torch.load(input_caption_tensor_path), torch.load(output_caption_tensor_path)

    image_dir_path = f'{DATA_DIR_PATH}/images'
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)

    with open(f'{DATA_DIR_PATH}/captions.txt', 'r') as caption_reader:
        image_captions = caption_reader.readlines()

    prev_image_name = None
    num_caption = len(image_captions) - 1
    assert num_caption % 5 == 0

    image_tensors = []
    input_caption_tensors = []
    output_caption_tensors = []
    for line in image_captions[1:]:
        image_name, caption = line.strip().split(',', 1)
        input_caption_tensor = torch.tensor([vocab[token.text] for token in tokenizer(caption.strip())])
        input_caption_tensors.append(torch.cat((torch.tensor([vocab['<SOS>']]), input_caption_tensor)))
        output_caption_tensors.append(torch.cat((input_caption_tensor, torch.tensor([vocab['<EOS>']]))))
        if image_name != prev_image_name:
            prev_image_name = image_name
            image = Image.open(os.path.join(image_dir_path, image_name))
            image_tensors.append(preprocess(image))

    processed_image_tensor = torch.stack(image_tensors)
    input_caption_tensor = nn.utils.rnn.pad_sequence(
        input_caption_tensors,
        batch_first=True,
        padding_value=vocab['<PAD>']
    )
    output_caption_tensor = nn.utils.rnn.pad_sequence(
        output_caption_tensors,
        batch_first=True,
        padding_value=vocab['<PAD>']
    )

    torch.save(processed_image_tensor, image_tensor_path)
    torch.save(input_caption_tensor, input_caption_tensor_path)
    torch.save(output_caption_tensor, output_caption_tensor_path)
    return processed_image_tensor, input_caption_tensor, output_caption_tensor
