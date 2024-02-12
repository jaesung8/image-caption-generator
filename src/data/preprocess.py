import glob
import os
import random

from PIL import Image
import torch
import torch.nn as nn 
from torchvision import transforms
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import pickle as pkl
from tqdm import tqdm

from src.constants import DATA_DIR_PATH


def preprocess_flickr(vocab):
    image_tensor_path = os.path.join(DATA_DIR_PATH, 'image_tensor.pt')
    input_caption_tensor_path = os.path.join(DATA_DIR_PATH, 'input_caption_tensor.pt')
    output_caption_tensor_path = os.path.join(DATA_DIR_PATH, 'output_caption_tensor.pt')
    if (
        os.path.isfile(image_tensor_path)
        and os.path.isfile(input_caption_tensor_path)
        and os.path.isfile(output_caption_tensor_path)
    ):
        return torch.load(image_tensor_path), torch.load(input_caption_tensor_path), torch.load(output_caption_tensor_path)

    preprocess = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(288),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # nlp = English()
    # tokenizer = Tokenizer(nlp.vocab)

    datasets = [
        'flickr8k',
        # 'flickr30k'
    ]
    image_tensors = []
    input_caption_tensors = []
    output_caption_tensors = []
    for datasets_name in datasets:
        image_dir_path = f'{DATA_DIR_PATH}/{datasets_name}/Images'
        with open(f'{DATA_DIR_PATH}/{datasets_name}/captions.txt', 'r') as caption_reader:
            image_captions = caption_reader.readlines()

        prev_image_name = None
        num_caption = len(image_captions) - 1
        assert num_caption % 5 == 0

        
        for line in image_captions[1:]:
            image_name, caption = line.strip().split(',', 1)
            # input_caption_tensor = torch.tensor([vocab[token.text] for token in tokenizer(caption.strip())])
            input_caption_tensor = torch.tensor([vocab[token] for token in caption.strip().replace('"', '').split()])
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

    print('save')
    torch.save(processed_image_tensor, image_tensor_path)
    torch.save(input_caption_tensor, input_caption_tensor_path)
    torch.save(output_caption_tensor, output_caption_tensor_path)
    print('save done')
    return processed_image_tensor, input_caption_tensor, output_caption_tensor


def preprocess_caption(vocab):
    # train_caption_path = os.path.join(DATA_DIR_PATH, 'train_captions.txt')
    # valid_caption_path = os.path.join(DATA_DIR_PATH, 'valid_captions.txt')
    # test_caption_path = os.path.join(DATA_DIR_PATH, 'test_captions.txt')
    train_caption_path = os.path.join(DATA_DIR_PATH, 'train_captions.pkl')
    valid_caption_path = os.path.join(DATA_DIR_PATH, 'valid_captions.pkl')
    test_caption_path = os.path.join(DATA_DIR_PATH, 'test_captions.pkl')
    if (
        os.path.isfile(train_caption_path)
        and os.path.isfile(valid_caption_path)
        and os.path.isfile(test_caption_path)
    ):
        # with open(train_caption_path, 'r') as caption_reader:
        #     train_captions = caption_reader.readlines()
        # with open(valid_caption_path, 'r') as caption_reader:
        #     valid_captions = caption_reader.readlines()
        # with open(test_caption_path, 'r') as caption_reader:
        #     test_captions = caption_reader.readlines()
        # return train_captions, valid_captions, test_captions
    
        with open(train_caption_path, 'rb') as caption_reader:
            train_captions = pkl.load(caption_reader)
        with open(valid_caption_path, 'rb') as caption_reader:
            valid_captions = pkl.load(caption_reader)
        with open(test_caption_path, 'rb') as caption_reader:
            test_captions = pkl.load(caption_reader)
        return train_captions, valid_captions, test_captions

    datasets = [
        'flickr8k',
        'flickr30k'
    ]
    results = []
    for datasets_name in datasets:
        with open(f'{DATA_DIR_PATH}/{datasets_name}/captions.txt', 'r') as caption_reader:
            image_captions = caption_reader.readlines()
        results = results + [f'{datasets_name}/Images/{line}' for line in image_captions[1:]]

    # random.shuffle(results)

    assert len(results) % 5 == 0
    assert len(results) % 10 == 0
    image_num = len(results) // 5
    shuffled_results = []
    print(len(results), image_num)
    shuffled_index = list(range(image_num))
    random.shuffle(shuffled_index)
    for image_index in tqdm(shuffled_index):
        all_captions = []
        encoded_cpations = []
        image_paths = []
        max_len = 0
        for line in results[image_index * 5:(image_index + 1) * 5]:
            image_path, caption = line.split(',', 1)
            image_path = os.path.join(DATA_DIR_PATH, image_path)
            encoded_cpation = [vocab[token] for token in caption.strip().replace('"', '').split()]
            encoded_cpation = [vocab['<SOS>']] + encoded_cpation + [vocab["<EOS>"]]
            if max_len < len(encoded_cpation):
                max_len = len(encoded_cpation)

            image_paths.append(image_path)
            encoded_cpations.append(encoded_cpation)
            all_captions.append(encoded_cpation)

        cur_chunk = []
        for i in range(5):
            if len(all_captions[i]) < max_len:
                for j in range(max_len - len(all_captions[i])):
                    all_captions[i].append(vocab['<PAD>'])
        for i in range(5):
            cur_chunk.append([
                image_paths[i],
                encoded_cpations[i],
                all_captions,
            ])
        shuffled_results.extend(cur_chunk)
    results = shuffled_results

    train_num, valid_num = int(0.8 * len(results)), int(0.1 * len(results))

    # with open(train_caption_path, 'w') as caption_writer:
    #     caption_writer.write(''.join(results[:train_num]))
    # with open(valid_caption_path, 'w') as caption_writer:
    #     caption_writer.write(''.join(results[train_num:train_num+valid_num]))
    # with open(test_caption_path, 'w') as caption_writer:
    #     caption_writer.write(''.join(results[train_num+valid_num:]))

    with open(train_caption_path, 'wb') as caption_writer:
        pkl.dump(results[:train_num], caption_writer)
    with open(valid_caption_path, 'wb') as caption_writer:
        pkl.dump(results[train_num:train_num+valid_num], caption_writer)
    with open(test_caption_path, 'wb') as caption_writer:
        pkl.dump(results[train_num+valid_num:], caption_writer)

    return results[:train_num], results[train_num:train_num+valid_num], results[train_num+valid_num:]
