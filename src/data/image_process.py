import glob

from PIL import Image
import torch
from torchvision import transforms

from src.constants import DATA_DIR_PATH


def preprocess_image():
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = glob.glob(f'{DATA_DIR_PATH}/images/*.jpg')
    processed_tensor = torch.empty(size=(len(image_paths), 1000))
    for i, image_path in enumerate(image_paths.sort()):
        image = Image.open(image_path)
        processed_tensor[i] = preprocess(image)

    processed_tensor.save(f'{DATA_DIR_PATH}/image_tensor.pt')
