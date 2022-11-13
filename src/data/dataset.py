import torch
from torch.utils.data import Dataset


class ImageCatpionDataset(Dataset):
    def __init__(self, images, captions, targets, start=0, end=1):
        image_num = len(images)
        start_index = image_num * start
        end_index = image_num * end
        self.images = images[start_index:end_index]
        self.captions = captions[start_index*5:end_index*5]
        self.targets = targets[start_index*5:end_index*5]

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
