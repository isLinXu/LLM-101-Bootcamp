import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def softmax(x, axis=-1):
    """数值稳定的softmax实现"""
    x_max = np.max(x, axis=axis, keepdims=True)
    x = x - x_max
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def positional_encoding_numpy(max_seq_length, d_model):
    """使用NumPy实现的位置编码"""
    pe = np.zeros((max_seq_length, d_model))
    
    for pos in range(max_seq_length):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    return pe

def plot_positional_encoding(pe, title="位置编码可视化"):
    """可视化位置编码"""
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(pe, cmap='RdBu')
    plt.xlabel('维度')
    plt.ylabel('位置')
    plt.colorbar()
    plt.title(title)
    plt.show()

def plot_attention_weights(weights, tokens_q=None, tokens_k=None, title="注意力权重"):
    """可视化注意力权重"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=tokens_k if tokens_k else "auto",
               yticklabels=tokens_q if tokens_q else "auto")
    plt.title(title)
    plt.xlabel('键位置')
    plt.ylabel('查询位置')
    plt.tight_layout()
    plt.show()

def scaled_dot_product_attention_numpy(q, k, v, mask=None):
    """使用NumPy实现的缩放点积注意力"""
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, v)
    
    return output, weights

class ScaledDotProductAttention(nn.Module):
    """PyTorch实现的缩放点积注意力"""
    def __init__(self, dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

class SelfAttention(nn.Module):
    """自注意力层"""
    def __init__(self, d_model, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        
    def forward(self, x, mask=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        output, weights = self.attention(q, k, v, mask)
        
        return output, weights

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, max_seq_length, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, d_model))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
    
    def forward(self, x):
        seq_length = x.size(1)
        return x + self.positional_encoding[:seq_length, :]

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    计算缩放点积注意力
    
    参数:
        q: 查询张量 shape == (..., seq_len_q, depth)
        k: 键张量 shape == (..., seq_len_k, depth)
        v: 值张量 shape == (..., seq_len_k, depth_v)
        mask: 可选掩码张量
        
    返回:
        输出张量和注意力权重
    """
    # 计算注意力分数
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    
    # 缩放注意力分数
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    
    # 添加掩码（如果提供）
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # softmax归一化权重
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    
    # 计算加权值
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    多头注意力层
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        """
        分割最后一个维度到 (num_heads, depth)
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # 缩放点积注意力
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        # (batch_size, num_heads, seq_len_q, depth)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        
        # (batch_size, seq_len_q, d_model)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        
        output = self.dense(concat_attention)
        
        return output, attention_weights

def plot_attention_weights(attention_weights, row_labels=None, col_labels=None, title=None):
    """
    可视化注意力权重
    
    参数:
        attention_weights: 注意力权重矩阵，形状为[seq_len_q, seq_len_k]
        row_labels: 行标签（查询位置）
        col_labels: 列标签（键位置）
        title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis')
    
    if row_labels is not None:
        plt.yticks(range(len(row_labels)), row_labels)
    if col_labels is not None:
        plt.xticks(range(len(col_labels)), col_labels)
    
    plt.xlabel('键位置')
    plt.ylabel('查询位置')
    plt.colorbar(label='注意力权重')
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    return plt 