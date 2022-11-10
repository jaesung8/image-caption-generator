import torch
from torch.utils.data import Dataset


class ImageCatpionDataset(Dataset):
    def __init__(self, images, captions, targets):
        self.images = images
        self.captions = captions
        self.targets = targets

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        # input_caption = torch.cat(
        #     torch.tensor(self.vocab['<START>']), self.captions[index]
        # )
        # output_caption = torch.cat(
        #     self.captions[index], torch.tensor(self.vocab['<END>'])
        # )
        return self.images[index//5], self.captions[index], self.targets[index]
