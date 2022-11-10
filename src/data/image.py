from PIL import Image
import torch
from torchvision import transforms


def preprocess_images(image_paths):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensors = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image_tensors.append(preprocess(image))
    return torch.tensor(image_tensors)
