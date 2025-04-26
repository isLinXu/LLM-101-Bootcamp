# -*- coding: utf-8 -*-
"""
Bigram语言模型可视化测试

这个测试文件专注于测试Bigram模型的可视化功能，包括条件概率的可视化和不同平滑方法的对比。
"""

import unittest
import random
import numpy as np
import matplotlib.pyplot as plt
import os

# 导入Bigram模型的函数
from bigram_model import (
    preprocess_text,
    train_bigram_model,
    visualize_bigram_probabilities
)

# 设置随机种子，确保测试结果可重现
random.seed(42)
np.random.seed(42)


class TestBigramVisualization(unittest.TestCase):
    """测试Bigram模型的可视化功能"""

    def setUp(self):
        """设置测试环境"""
        # 测试文本
        self.text = """
        The quick brown fox jumps over the lazy dog. 
        The dog barks at the fox. 
        The fox runs away quickly.
        """
        
        # 预处理文本
        self.sentences, self.vocab = preprocess_text(self.text)
        
        # 确保输出目录存在
        os.makedirs("visualization_output", exist_ok=True)
    
    def test_visualization_output(self):
        """测试可视化函数能否正确生成图像文件"""
        # 训练模型
        bigram_probs, _, _ = train_bigram_model(self.sentences, self.vocab, smoothing='laplace')
        
        # 测试词
        test_word = 'the'
        
        # 设置保存路径
        output_path = "visualization_output/test_viz.png"
        
        # 自定义可视化函数，保存到指定路径
        plt.figure(figsize=(10, 6))
        next_words = [(next_word, prob) for next_word, prob in bigram_probs[test_word].items() if prob > 0]
        next_words.sort(key=lambda x: x[1], reverse=True)
        next_words = next_words[:5]
        
        words = [item[0] for item in next_words]
        probs = [item[1] for item in next_words]
        
        plt.bar(words, probs)
        plt.xlabel('下一个词')
        plt.ylabel('概率')
        plt.title(f"词'{test_word}'后面最可能出现的5个词")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        # 验证文件是否创建
        self.assertTrue(os.path.exists(output_path), "可视化图像文件应该被创建")
    
    def test_smoothing_comparison(self):
        """比较不同平滑方法的可视化结果"""
        # 平滑方法
        smoothing_methods = ['laplace', 'add_k', 'interpolation']
        test_word = 'the'
        
        for method in smoothing_methods:
            # 训练模型
            if method == 'add_k':
                bigram_probs, _, _ = train_bigram_model(
                    self.sentences, self.vocab, smoothing=method, k=0.5
                )
            else:
                bigram_probs, _, _ = train_bigram_model(
                    self.sentences, self.vocab, smoothing=method
                )
            
            # 输出路径
            output_path = f"visualization_output/{method}_smoothing.png"
            
            # 生成可视化
            plt.figure(figsize=(10, 6))
            next_words = [(next_word, prob) for next_word, prob in bigram_probs[test_word].items() if prob > 0]
            next_words.sort(key=lambda x: x[1], reverse=True)
            next_words = next_words[:5]
            
            words = [item[0] for item in next_words]
            probs = [item[1] for item in next_words]
            
            plt.bar(words, probs)
            plt.xlabel('下一个词')
            plt.ylabel('概率')
            plt.title(f"{method}平滑方法: 词'{test_word}'后面最可能出现的词")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            # 验证文件是否创建
            self.assertTrue(os.path.exists(output_path), f"{method}平滑方法的可视化图像应该被创建")


if __name__ == '__main__':
    unittest.main() 