import torch
from torch.utils.data import Dataset


class ImageCatpionDataset(Dataset):
    def __init__(self, images, captions):
        self.images = images
        self.captions = captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        images = self.images[index].tile((5,))
        return images, self.captions[index*5:(index+1)*5]
