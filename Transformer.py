from torch import nn

from MultiHeadAttention import MultiHeadAttention
from Normalization import PreNorm, PostNorm
from Residual import Residual
from FeedForward import FeedForward


class SelfTransformer(nn.Module):
    def __init__(self, dim, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PostNorm(dim, MultiHeadAttention(dim))),
                Residual(PostNorm(dim, FeedForward(dim))),
            ]))

    def forward(self, x):
        'x: question'
        for att, feed in self.layers:
            x = att(x)
            x = feed(x)
        return x


class CrossTransformer(nn.Module):
    def __init__(self, dim, dim_cross, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PostNorm(dim, MultiHeadAttention(dim))),
                Residual(PostNorm(dim, MultiHeadAttention(dim, dim_cross))),
                Residual(PostNorm(dim, FeedForward(dim))),
            ]))

    def forward(self, x, crossContext):
        for selfAtt, crossAtt, feed in self.layers:
            x = selfAtt(x)
            x = crossAtt(x, crossContext)
            x = feed(x)
        return x


class Transformer(nn.Module):

    def __init__(self, input_dim, output_dim, depth=3):
        super().__init__()
        self.selfTransformer = SelfTransformer(input_dim, depth=depth)
        self.crossTransformer = CrossTransformer(output_dim, input_dim, depth=depth)

    def forward(self, input, output):
        x = self.selfTransformer(input)
        x = self.crossTransformer(output, x)
        return x
