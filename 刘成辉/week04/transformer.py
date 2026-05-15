# 尝试用pytorch实现一个transformer层。
import math

import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    第二、三阶段：多头注意力机制 (灵魂)
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        # 保证 维度 可以被整除
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. 线性变换得到 Q, K, V
        # q:
        # view:[batch_size, seq_length, d_model] -> [batch_size, seq_length, self.n_heads, self.d_k]
        # transpose:-> [batch_size, self.n_heads, seq_length, self.d_k]
        Q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算 Scaled Dot-Product Attention: softmax(QK^T / sqrt(dk)) * V [cite: 38]
        # [batch_size, self.n_heads, seq_length, self.d_k] * [batch_size, self.n_heads, self.d_k, seq_length]
        scores = Q @ K.transpose(-2, -1)/math.sqrt(self.d_k)
        # 应用 Mask (如果提供)
        if mask is not None:
            # 这里的 mask 主要是 Padding Mask
            # 将 mask 为 0 的位置填入一个极小值，使其在 softmax 后接近 0
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = nn.functional.softmax(scores, dim=-1)
        # 应用 Dropout
        attention = self.dropout(attention)

        out_put = (attention @ V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(out_put)


class TransformerBlock(nn.Module):
    """
    第四阶段：Encoder Block (模块化)
    包含：MHA + Add&Norm + FFN + Add&Norm
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        # 多头注意力子层
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络子层 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
            # 实现两层线性层，中间夹一个激活函数（BERT常用GELU）
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        qkv = self.norm1(x)
        # 实现残差连接：x = x + self.attention(x) -> norm(x)
        x = self.attention(qkv, qkv, qkv, mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.ffn(self.norm2(x))
        x = residual + x
        return x