# Mixture of Experts (MoE) Network for Watermark Decoder
# *************************************************************************
# This file may have been modified by Bytedance Inc. ("Bytedance Inc.'s Mo-
# difications"). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
import torch
from torch import nn
import torch.nn.functional as F
from .ConvNet import ConvBNRelu


class Expert(nn.Module):
    """
    Single expert network for MoE
    """
    def __init__(self, in_channels, out_channels, expert_blocks=2):
        super(Expert, self).__init__()
        
        layers = []
        layers.append(ConvBNRelu(in_channels, out_channels))
        for _ in range(expert_blocks - 1):
            layers.append(ConvBNRelu(out_channels, out_channels))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class GatingNetwork(nn.Module):
    """
    Gating network to determine the weight of each expert
    """
    def __init__(self, in_channels, num_experts):
        super(GatingNetwork, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]
        pooled = self.global_pool(x).view(B, -1)  # (B, C)
        weights = self.fc(pooled)  # (B, num_experts)
        return weights


class MoEBlock(nn.Module):
    """
    Mixture of Experts Block
    Combines multiple expert networks with a gating mechanism
    """
    def __init__(self, in_channels, out_channels, num_experts=4, expert_blocks=2):
        super(MoEBlock, self).__init__()
        
        self.num_experts = num_experts
        
        # Create multiple experts
        self.experts = nn.ModuleList([
            Expert(in_channels, out_channels, expert_blocks) 
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating = GatingNetwork(in_channels, num_experts)
    
    def forward(self, x):
        # Get gating weights
        gating_weights = self.gating(x)  # (B, num_experts)
        
        # Get outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # Stack expert outputs: (num_experts, B, C, H, W)
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # Reshape gating weights for broadcasting: (num_experts, B, 1, 1, 1)
        gating_weights = gating_weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Weighted combination of expert outputs
        output = (expert_outputs * gating_weights).sum(dim=0)
        
        return output


class MoENet(nn.Module):
    """
    Network composed of multiple MoE blocks
    """
    def __init__(self, in_channels, out_channels, blocks, num_experts=4, expert_blocks=2):
        super(MoENet, self).__init__()
        
        layers = []
        if blocks > 0:
            layers.append(MoEBlock(in_channels, out_channels, num_experts, expert_blocks))
            for _ in range(blocks - 1):
                layers.append(MoEBlock(out_channels, out_channels, num_experts, expert_blocks))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
