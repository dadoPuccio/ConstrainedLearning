import torch.nn as nn
from utils.options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
# from torch.utils.checkpoint import checkpoint
class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(channels_in=3, channels_out=config.discriminator_channels)]
        for _ in range(config.discriminator_blocks-1):
            layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1))) # type: ignore
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(config.discriminator_channels, 1)

    def forward(self, image):
        X = self.before_linear(image)
        # X = checkpoint(self.before_linear, image, use_reentrant=False)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)  # type: ignore
        X = self.linear(X)
        # X = checkpoint(self.linear, X)
        # X = torch.sigmoid(X)
        return X