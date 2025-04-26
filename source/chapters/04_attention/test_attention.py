import numpy as np
import torch
import unittest
from attention_utils import (
    softmax,
    positional_encoding_numpy,
    scaled_dot_product_attention_numpy,
    SelfAttention,
    PositionalEncoding,
    scaled_dot_product_attention,
    MultiHeadAttention
)

class TestAttentionMechanism(unittest.TestCase):
    
    def test_softmax(self):
        """测试softmax函数的正确性"""
        # 测试向量
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = softmax(x)
        # 验证和为1
        self.assertAlmostEqual(np.sum(result), 1.0)
        # 验证排序不变
        self.assertTrue(np.all(np.diff(result) > 0))
        
        # 测试数值稳定性
        x_large = np.array([1000.0, 2000.0, 3000.0])
        result_large = softmax(x_large)
        self.assertAlmostEqual(np.sum(result_large), 1.0)
        self.assertFalse(np.any(np.isnan(result_large)))
        
        # 测试矩阵
        x_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result_matrix = softmax(x_matrix, axis=1)
        self.assertAlmostEqual(np.sum(result_matrix[0]), 1.0)
        self.assertAlmostEqual(np.sum(result_matrix[1]), 1.0)
    
    def test_positional_encoding(self):
        """测试位置编码的正确性"""
        max_seq_length = 10
        d_model = 8
        pe = positional_encoding_numpy(max_seq_length, d_model)
        
        # 验证形状
        self.assertEqual(pe.shape, (max_seq_length, d_model))
        
        # 验证周期性
        # 对于相同的维度，位置0和位置10000应该有相似的值
        small_pos = 0
        large_pos = 100
        small_pe = positional_encoding_numpy(1, d_model)[small_pos]
        large_pe = positional_encoding_numpy(large_pos + 1, d_model)[large_pos]
        for i in range(0, d_model, 2):
            wavelength = 10000 ** (i / d_model)
            # 检查是否在同一周期中的相似位置
            small_angle = small_pos / wavelength
            large_angle = large_pos / wavelength
            angle_diff = (large_angle - small_angle) % (2 * np.pi)
            if angle_diff < 1e-6 or abs(angle_diff - 2 * np.pi) < 1e-6:
                self.assertAlmostEqual(small_pe[i], large_pe[i], places=5)
    
    def test_scaled_dot_product_attention(self):
        """测试缩放点积注意力的正确性"""
        batch_size = 2
        seq_len_q = 3
        seq_len_k = 4
        depth = 8
        
        q = torch.randn(batch_size, seq_len_q, depth)
        k = torch.randn(batch_size, seq_len_k, depth)
        v = torch.randn(batch_size, seq_len_k, depth)
        
        # 运行注意力函数
        output, attention_weights = scaled_dot_product_attention(q, k, v)
        
        # 验证输出形状
        self.assertEqual(output.shape, (batch_size, seq_len_q, depth))
        self.assertEqual(attention_weights.shape, (batch_size, seq_len_q, seq_len_k))
        
        # 验证注意力权重的每一行之和为1（softmax特性）
        row_sums = attention_weights.sum(dim=-1)
        expected_sums = torch.ones_like(row_sums)
        self.assertTrue(torch.allclose(row_sums, expected_sums, atol=1e-6))
    
    def test_mask_in_attention(self):
        """测试在注意力中使用掩码"""
        batch_size = 2
        seq_len = 4
        d_model = 4
        
        # 创建测试数据
        np.random.seed(42)
        q = np.random.randn(batch_size, seq_len, d_model)
        k = q  # 自注意力
        v = q
        
        # 创建一个上三角掩码（防止关注未来位置）
        mask = np.triu(np.ones((batch_size, seq_len, seq_len)), k=1) == 0
        
        # 计算带掩码的注意力
        output, weights = scaled_dot_product_attention_numpy(q, k, v, mask)
        
        # 验证掩码位置的权重接近于0
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    if j > i:  # 未来位置
                        self.assertAlmostEqual(weights[b, i, j], 0.0, places=5)
    
    def test_pytorch_self_attention(self):
        """测试PyTorch实现的自注意力层"""
        batch_size = 2
        seq_len = 3
        d_model = 4
        
        # 创建测试数据
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 创建自注意力层
        self_attn = SelfAttention(d_model)
        
        # 计算自注意力
        output, weights = self_attn(x)
        
        # 验证形状
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertEqual(weights.shape, (batch_size, seq_len, seq_len))
        
        # 验证权重和为1
        weights_sum = torch.sum(weights, dim=-1)
        for b in range(batch_size):
            for i in range(seq_len):
                self.assertAlmostEqual(weights_sum[b, i].item(), 1.0, places=5)
    
    def test_positional_encoding_module(self):
        """测试PyTorch实现的位置编码模块"""
        batch_size = 2
        seq_len = 5
        d_model = 8
        
        # 创建测试数据
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 创建位置编码模块
        pos_encoder = PositionalEncoding(d_model)
        
        # 应用位置编码
        encoded = pos_encoder(x)
        
        # 验证形状
        self.assertEqual(encoded.shape, (batch_size, seq_len, d_model))
        
        # 验证位置编码已添加
        self.assertFalse(torch.allclose(x, encoded))

    def test_attention_with_mask(self):
        # 创建测试数据
        batch_size = 2
        seq_len = 4
        depth = 8
        
        q = torch.randn(batch_size, seq_len, depth)
        k = torch.randn(batch_size, seq_len, depth)
        v = torch.randn(batch_size, seq_len, depth)
        
        # 创建一个掩码（上三角矩阵）
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0)
        
        # 运行带掩码的注意力函数
        output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # 验证被掩码的位置的权重接近于0
        masked_weights = attention_weights.masked_select(mask.bool())
        self.assertTrue(torch.all(masked_weights < 1e-6))
    
    def test_multi_head_attention(self):
        # 创建测试数据
        batch_size = 2
        seq_len = 5
        d_model = 64
        num_heads = 8
        
        # 实例化多头注意力层
        mha = MultiHeadAttention(d_model, num_heads)
        
        # 创建输入
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 自注意力：使用相同的输入作为查询、键和值
        output, attention_weights = mha(x, x, x)
        
        # 验证输出形状
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertEqual(attention_weights.shape, (batch_size, num_heads, seq_len, seq_len))
        
        # 验证注意力权重的每一行之和为1
        row_sums = attention_weights.sum(dim=-1)
        expected_sums = torch.ones_like(row_sums)
        self.assertTrue(torch.allclose(row_sums, expected_sums, atol=1e-6))

if __name__ == '__main__':
    unittest.main() 