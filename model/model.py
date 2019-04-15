import torch.nn as nn
from base import BaseModel
from torchvision.models import vgg16_bn


class VGG16BN(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = vgg16_bn(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 128), nn.ReLU(True), nn.Dropout(), nn.Linear(128, num_classes)
        )
        # freeze all layers of feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
        self._initialize_classifier_weights()

    def _initialize_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
