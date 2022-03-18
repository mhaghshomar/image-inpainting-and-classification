import torch


class Block(torch.nn.Module):
    def __init__(self, features, kernel_size=(3, 3), activation="ReLU"):
        super().__init__()
        padding = (
            kernel_size // 2
            if not isinstance(kernel_size, tuple)
            else kernel_size[0] // 2,
            kernel_size[1] // 2,
        )
        self.conv1 = torch.nn.LazyConv2d(
            out_channels=features, kernel_size=kernel_size, padding=padding
        )
        self.relu = getattr(torch.nn, activation)()
        self.conv2 = torch.nn.LazyConv2d(
            out_channels=features, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(torch.nn.Module):
    def __init__(self, blocks=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = torch.nn.ModuleList(
            [Block(features=block_size) for block_size in blocks]
        )
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x, return_features_list: bool = False):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            if return_features_list:
                features.append(x)
            x = self.pool(x)
        return features if return_features_list else x


class Decoder(torch.nn.Module):
    def __init__(self, blocks=(1024, 512, 256, 128, 64), kernel_size=2, stride=2):
        super().__init__()
        self.blocks = blocks
        self.upconvs = torch.nn.ModuleList(
            [
                torch.nn.LazyConvTranspose2d(
                    out_channels=block, kernel_size=kernel_size, stride=stride
                )
                for block in blocks
            ]
        )
        self.dec_blocks = torch.nn.ModuleList(
            [Block(features=block) for block in blocks]
        )

    def forward(self, features):
        features = features[::-1]
        for i in range(len(self.blocks) - 1):
            x = self.upconvs[i](features[i])
            x = torch.cat([x, features[i + 1]], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet(torch.nn.Module):
    def __init__(self, in_channels=1, features=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.encoder = Encoder(blocks=features)
        self.decoder = Decoder(blocks=features[::-1])
        self.head = torch.nn.LazyConv2d(in_channels, kernel_size=1)

    def forward(self, x):
        return self.head(self.decoder(self.encoder(x, return_features_list=True))) + x


