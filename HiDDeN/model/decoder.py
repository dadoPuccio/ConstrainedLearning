import torch.nn as nn
from utils.options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
# from torch.utils.checkpoint import checkpoint

class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, config: HiDDenConfiguration):

        super(Decoder, self).__init__()
        self.channels = config.decoder_channels

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, config.message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1))) # type: ignore
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(config.message_length, config.message_length)

    def forward(self, image_with_wm):

        x = self.layers(image_with_wm)
        # x = checkpoint(self.layers, image_with_wm, use_reentrant=False)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2) # type: ignore
        x = self.linear(x)
        # x = checkpoint(self.linear, x, use_reentrant=False)
        return x
