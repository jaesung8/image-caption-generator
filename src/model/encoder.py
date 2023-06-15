import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        efficientnet = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.efficientnet = efficientnet
        # self.efficientnet._forward_impl = self.efficientnet.features
        # self.efficientnet.forward = self.efficientnet.features
        # self.efficientnet.classifier = None
        self.fc = nn.Linear(1000, embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.9)

    def forward(self, images):
        features = self.efficientnet(images)
        # [64, 1408, 9, 9]
        # features = self.efficientnet.features(images)
        outputs = self.fc(features)
        # outputs = self.dropout(self.relu(features))
        return outputs