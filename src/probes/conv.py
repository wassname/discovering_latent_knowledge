import torch
import torch.nn as nn

from .pl_ranking import PLRanking

class ConvProbe(nn.Module):
    def __init__(self, c_in, depth=0, hs=16, dropout=0):
        super().__init__()

        layers = [
            nn.BatchNorm1d(c_in, affine=False),  # this will normalise the inputs
            nn.Dropout1d(dropout),
            
            nn.Conv1d(c_in, hs*(depth+1), kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm1d(hs*(depth+1)),
        ]
        for i in range(depth):
            layers += [
                nn.Conv1d(hs*(depth-i+1), hs*(depth-i), 2),
                nn.ReLU(),
                nn.BatchNorm1d(hs*(depth-i)),
                
            ]
        layers += [nn.AdaptiveAvgPool1d(1)]
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hs, hs), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hs, 1)            
        )

    def forward(self, x):
        h = self.net(x)
        # print(1, h.shape)
        h = h.squeeze(-1)
        # print(1, h.shape)
        return self.head(h)


class PLConvProbe(PLRanking):
    def __init__(self, c_in, *args, depth=1, dropout=0, hs=16, **kwargs):
        super().__init__(c_in, *args, depth=depth, dropout=dropout, hs=hs, **kwargs)
        self.probe = ConvProbe(c_in, depth=depth, dropout=dropout, hs=hs)
