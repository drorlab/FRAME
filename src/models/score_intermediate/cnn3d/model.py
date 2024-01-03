import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CNN3D_Model(nn.Module):

    def __init__(self, in_dim, spatial_size,
                 conv_drop_rate, fc_drop_rate,
                 batch_norm, dropout,
                 num_conv, conv_kernel_size,
                 fc_filters, **kwargs):
        super().__init__()

        ### Define network layers
        conv_filters = [32 * (2**n) for n in range(num_conv)]
        max_pool_positions = [0, 1]*int((num_conv+1)/2)
        max_pool_sizes = [2]*num_conv
        max_pool_strides = [2]*num_conv

        layers = []
        in_channels = in_dim

        if batch_norm:
            layers.append(nn.BatchNorm3d(in_channels))

        # Convs
        for i in range(len(conv_filters)):
            layers.extend([
                nn.Conv3d(in_channels, conv_filters[i],
                          kernel_size=conv_kernel_size,
                          bias=True),
                nn.ReLU()
                ])
            spatial_size -= (conv_kernel_size - 1)
            if max_pool_positions[i]:
                layers.append(nn.MaxPool3d(max_pool_sizes[i], max_pool_strides[i]))
                spatial_size = int(np.floor((spatial_size - (max_pool_sizes[i]-1) - 1)/max_pool_strides[i] + 1))
            if batch_norm:
                layers.append(nn.BatchNorm3d(conv_filters[i]))
            if dropout:
                layers.append(nn.Dropout(conv_drop_rate))
            in_channels = conv_filters[i]

        layers.append(nn.Flatten())
        in_features = in_channels * (spatial_size**3)
        # FC layers
        for units in fc_filters:
            layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU()
                ])
            if batch_norm:
                layers.append(nn.BatchNorm3d(units))
            if dropout:
                layers.append(nn.Dropout(fc_drop_rate))
            in_features = units

        # Final FC layer
        layers.append(nn.Linear(in_features, 1))

        self.net = nn.Sequential(*layers)
        print(self.net)

    def forward(self, d):
        out = self.net(d['x']).view(-1)
        return out
