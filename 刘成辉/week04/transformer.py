# 尝试用pytorch实现一个transformer层。
import math

import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    高效的多头注意力机制 (无显式复制)
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_qkv = nn.Linear(d_model, 3 * d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape

        # 融合投影 + chunk 物理切块
        qkv_fused = self.w_qkv(x)
        q, k, v = qkv_fused.chunk(3, dim=-1)

        # 多头拆分 [B, H, L, d_k]
        q = q.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)

        # 矩阵乘法计算注意力分数
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = nn.functional.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 头拼接还原
        out_put = (attention @ v).transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        return self.fc_out(out_put)


class TransformerBlock(nn.Module):
    """
    模块化的 Transformer Encoder 层 (采用现代大模型标准的 Pre-LN 结构)
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
            nn.Dropout(dropout),  # ✨ 补充规范：FFN 内部的 Dropout 极为重要
            nn.Linear(d_ff, d_model)
            # 实现两层线性层，中间夹一个激活函数（BERT常用GELU）
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- 第一子层：MHA + Add & Norm (Pre-Norm 结构) ---
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)  # 残差相加

        # --- 第二子层：FFN + Add & Norm ---
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)  # ✨ 补充规范：FFN 输出后同样做一次 Dropout 再残差

        return x