import os

os.environ["DGLBACKEND"] = "pytorch"
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.data
from dgl.nn import GATv2Conv, GlobalAttentionPooling

from myprojectutils import MyUtils


class MyGATv2(nn.Module):

    def __init__(
        self,
        h_dim,
        num_classes,
        num_heads,
        feat_dropout,
        attn_dropout,
        layers,
        in_dimN=15,
    ) -> None:
        super(MyGATv2, self).__init__()

        self.convsN = nn.Sequential(
            GATv2Conv(
                in_dimN,
                h_dim,
                num_heads=num_heads,
                feat_drop=feat_dropout,
                attn_drop=attn_dropout,
                residual=True,
                activation=F.relu,
                allow_zero_in_degree=True,
            )
        )

        for i in range(1, layers):
            self.convsN.add_module(
                str(i),
                GATv2Conv(
                    h_dim * num_heads,
                    h_dim,
                    num_heads=num_heads,
                    feat_drop=feat_dropout,
                    attn_drop=attn_dropout,
                    residual=True,
                    activation=F.relu,
                    allow_zero_in_degree=True,
                ),
            )

        self.gPool = GlobalAttentionPooling(nn.Linear(h_dim * num_heads, 1))
        self.dropout = nn.Dropout(feat_dropout)

        self.linear = nn.Linear(h_dim * num_heads, 2 * h_dim)
        self.classify = nn.Linear(2 * h_dim, num_classes)

    def forward(self, g):
        hN = MyUtils.get_feature(g)

        hg = None
        for i in range(len(self.convsN)):
            hN = F.relu(self.convsN[i](g, hN).flatten(1))

            with g.local_scope():
                g.ndata["h"] = hN
                newhN = dgl.softmax_nodes(g, "h")

                if hg == None:
                    hg = self.gPool(g, newhN)
                else:
                    hg = hg + self.gPool(g, newhN)

        return self.classify(F.relu(self.dropout(self.linear(hg))))
