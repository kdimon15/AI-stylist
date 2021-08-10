import timm
from torch import nn
from Config import CFG


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=CFG.pretrained)
        self.model.classifier = nn.Sequential()

    def forward(self, x):
        output = self.model(x)
        return output
