import torch
import torch.nn as nn
from model.resnet import resnet50, resnet10, resnet18
from model.graphunet import GraphUNet, GraphNet
from model.modulated_gcn import ModulatedGCN
import numpy as np


class MyNet(nn.Module):
    def __init__(self, adj, block) -> None:
        super(MyNet, self).__init__()
        self.resnet = resnet18(pretrained=False, num_classes=17*2)
        self.graph = GraphNet(in_features=514, out_features=2)
        self.MGCN = ModulatedGCN(adj, 384, num_layers=block, p_dropout=0, nodes_group=None)

    def forward(self, x):
        N = x.shape[0]
        points2D_init, features = self.resnet(x)
        features = features.unsqueeze(1).repeat(1, 17, 1)
        in_features = torch.cat([points2D_init, features], dim=2)
        out_2d = self.graph(in_features)
        out_2d_gcn = out_2d.view(N, -1, 17, 2, 1).permute(0, 3, 1, 2, 4)
        out_3d = self.MGCN(out_2d_gcn)

        return out_3d, out_2d, points2D_init

