from torch import nn, einsum
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, d_k=64, dropout=0.1) -> None:
        super().__init__()
        if not self.ifExist(context_dim):
            context_dim = query_dim
        self.heads = heads
        self.scale = d_k ** -0.5
        d_model = d_k * heads
        self.to_q = nn.Linear(query_dim, d_model, bias=False)
        self.to_kv = nn.Linear(context_dim, 2 * d_model, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_model, query_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        if not self.ifExist(context):
            context = x

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda mat: rearrange(mat, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        qkT = einsum('b n d, b m d->b n m', q, k) * self.scale
        attention = qkT.softmax(dim=-1)
        attention = einsum('b n m, b m d->b n d', attention, v)
        attention = rearrange(attention, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(attention)

    @staticmethod
    def ifExist(var):
        if var is None:
            return False
        else:
            return True
