from unittest import TestCase
import torch
from torch import nn
from einops import repeat

from Transformer import Transformer


class TestTransformer(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.question_dim = 16
        self.latent_dim = 8
        self.question = nn.Parameter(torch.randn((self.latent_dim, self.question_dim)))

        self.content_dim = 256
        self.content = nn.Parameter(torch.randn((self.latent_dim, self.content_dim)))

        self.batch_size = 1
        self.question = repeat(self.question, 'l d -> b l d', b=self.batch_size)
        self.content = repeat(self.content, 'l d -> b l d', b=self.batch_size)
        self.transformer = Transformer(self.question_dim, self.content_dim)

    def test_transformer(self):
        out = self.transformer(self.question, self.content)
        pass
