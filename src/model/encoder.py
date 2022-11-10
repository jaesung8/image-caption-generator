import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights


class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features[0]))
