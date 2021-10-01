from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.GELU(),
            nn.Linear(dim * 3, dim)
        )

    def forward(self, x):
        return self.net(x)
