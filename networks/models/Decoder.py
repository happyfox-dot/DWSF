# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/Decoder.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from ..basic_blocks.ConvNet import ConvBNRelu
from ..basic_blocks.SENet import SENet, SENet_decoder
from ..basic_blocks.MoENet import MoEBlock
from torch import nn


class Decoder(nn.Module):
    """
    Decoder for proposed method with optional MoE architecture
    """
    def __init__(self, message_length=30, decoder_channels=64, in_channel=3, use_moe=False, num_experts=4):

        super(Decoder, self).__init__()
        self.channels = decoder_channels
        self.use_moe = use_moe

        if use_moe:
            # MoE-enhanced decoder with mixture of experts blocks
            self.layers = nn.Sequential(
                ConvBNRelu(in_channel, self.channels),
                MoEBlock(self.channels, self.channels, num_experts=num_experts, expert_blocks=2),
                SENet(self.channels, self.channels, blocks=2),
                SENet_decoder(self.channels, self.channels, blocks=2, drop_rate2=2),
                MoEBlock(self.channels*2, self.channels, num_experts=num_experts, expert_blocks=2),
                SENet(self.channels, self.channels, blocks=2),
                SENet_decoder(self.channels, self.channels, blocks=2, drop_rate2=2),
                MoEBlock(self.channels*2, self.channels, num_experts=num_experts, expert_blocks=2),
                SENet(self.channels, self.channels, blocks=2),
                SENet_decoder(self.channels, self.channels, blocks=2, drop_rate2=2),
                MoEBlock(self.channels*2, self.channels, num_experts=num_experts, expert_blocks=2),
                SENet(self.channels, self.channels, blocks=2),
                SENet_decoder(self.channels, self.channels, blocks=2, drop_rate2=2),
                nn.Conv2d(self.channels*2, 1, kernel_size=1)
            )
        else:
            # Original decoder architecture
            self.layers = nn.Sequential(
                ConvBNRelu(in_channel, self.channels),
                SENet(self.channels, self.channels, blocks=4),
                SENet_decoder(self.channels, self.channels, blocks=2, drop_rate2=2),
                SENet(self.channels*2, self.channels, blocks=4),
                SENet_decoder(self.channels, self.channels, blocks=2, drop_rate2=2),
                SENet(self.channels*2, self.channels, blocks=4),
                SENet_decoder(self.channels, self.channels, blocks=2, drop_rate2=2),
                SENet(self.channels*2, self.channels, blocks=4),
                SENet_decoder(self.channels, self.channels, blocks=2, drop_rate2=2),
                nn.Conv2d(self.channels*2, 1, kernel_size=1)
            )

        self.linear = nn.Linear(self.channels, message_length)

        self.activation = nn.ReLU(True)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.activation(x)
        return x
