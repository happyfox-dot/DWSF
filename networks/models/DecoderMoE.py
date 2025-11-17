# MoE-based Decoder for watermark extraction
# *************************************************************************
# This file may have been modified by Bytedance Inc. ("Bytedance Inc.'s Mo-
# difications"). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from ..basic_blocks.ConvNet import ConvBNRelu
from ..basic_blocks.SENet import SENet, SENet_decoder
from ..basic_blocks.MoENet import MoENet, MoEBlock
from torch import nn


class DecoderMoE(nn.Module):
    """
    MoE-based Decoder for proposed method
    Uses Mixture of Experts architecture for improved feature extraction
    """
    def __init__(self, message_length=30, decoder_channels=64, in_channel=3, num_experts=4):

        super(DecoderMoE, self).__init__()
        self.channels = decoder_channels
        self.num_experts = num_experts

        # Initial convolution
        self.initial_conv = ConvBNRelu(in_channel, self.channels)
        
        # MoE-based feature extraction layers
        # Replace some SENet blocks with MoE blocks for adaptive feature learning
        self.layers = nn.Sequential(
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

        self.linear = nn.Linear(self.channels, message_length)

        self.activation = nn.ReLU(True)

    def forward(self, image_with_wm):
        x = self.initial_conv(image_with_wm)
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.activation(x)
        return x
