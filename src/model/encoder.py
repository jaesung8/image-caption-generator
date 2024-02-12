import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet50, ResNet50_Weights


class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet = efficientnet
        # self.efficientnet._forward_impl = self.efficientnet.features
        # self.efficientnet.forward = self.efficientnet.features
        # self.efficientnet.classifier = None
        self.fc = nn.Linear(62720, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        # features = self.efficientnet(images)
        with torch.no_grad():
            features = self.efficientnet.features(images)
            # [96, 1280, 7, 7]
            features = torch.flatten(features, 1)
        outputs = self.fc(features)
        outputs = self.relu(outputs)
        return outputs
    

class AttentionEncoder(nn.Module):
    def __init__(self):
        super(AttentionEncoder, self).__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    def forward(self, images):
        with torch.no_grad():
            x = self.encoder.conv1(images)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)

            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            x = self.encoder.layer3(x)
            features = self.encoder.layer4(x)
            # [96, 1280, 7, 7]

            features = features.permute(0, 2, 3, 1)
            features = features.view(features.size(0), -1, features.size(-1))
        return features
