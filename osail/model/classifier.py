import torch
from .unet import UNet

class Classifier(torch.nn.Module):
    def __init__(self, unet_checkpoipnt_path: str=None, unet_params: dict = None, num_classes: int = 2):
        super().__init__()
        unet = UNet(**(unet_params or dict()))
        if unet_checkpoipnt_path:
            unet.load_state_dict(torch.load(unet_checkpoipnt_path))
        self.encoder = unet.encoder
        self.head = torch.nn.LazyLinear(out_features=num_classes, bias=True)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        return self.softmax(self.head(self.encoder(x).reshape(x.shape[0], -1)))

