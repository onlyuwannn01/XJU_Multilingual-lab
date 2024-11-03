
import torch
from torch import nn
import torch.functional as F
import math


# X = torch.randn(50, 100, 1024)

# d_model = 1024
# n_head = 8

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Multi_head_Attention(nn.Module):
    def __init__(self,d_model,n_head):
        super(Multi_head_Attention,self).__init__()
        self.n_head = n_head
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_combine = nn.Linear(d_model, d_model)
        self.softMax = nn.Softmax(dim=-1)


    def forward(self,q,k,v):

        batch,time,dimension = q.shape
        n_d = self.d_model // self.n_head

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(n_d)

        mask = torch.tril(torch.ones(time, time, dtype=bool)).to(DEVICE)
        score = score.masked_fill(mask == 0, float("-inf")).to(DEVICE)
        score = self.softMax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)

        output = self.w_combine(score)
        return output

# attention = Multi_head_Attention(d_model, n_head)

# output = attention(X, X, X)

# print(output, output.shape)
