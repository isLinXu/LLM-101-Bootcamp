# -*- coding: utf-8 -*-
"""
Micrograd示例

这个脚本展示了如何使用Micrograd构建和训练神经网络。
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from micrograd.engine import Value
from micrograd.nn import MLP

# 设置随机种子，确保结果可重现
random.seed(42)
np.random.seed(42)

def generate_dataset():
    """生成一个简单的二分类数据集"""
    # 生成螺旋数据
    n_points = 100
    n_classes = 2
    X = []
    y = []
    for i in range(n_classes):
        for j in range(n_points):
            r = j / n_points * 5
            t = 1.25 * j / n_points * 2 * np.pi + i * np.pi
            X.append([r * np.sin(t), r * np.cos(t)])
            y.append(1.0 if i == 0 else -1.0)
    return X, y

def plot_data(X, y):
    """可视化数据集"""
    plt.figure(figsize=(8, 8))
    plt.scatter([x[0] for i, x in enumerate(X) if y[i] > 0], 
                [x[1] for i, x in enumerate(X) if y[i] > 0], 
                c='r', marker='o', label='类别 1')
    plt.scatter([x[0] for i, x in enumerate(X) if y[i] < 0], 
                [x[1] for i, x in enumerate(X) if y[i] < 0], 
                c='b', marker='x', label='类别 2')
    plt.legend()
    plt.title('螺旋数据集')
    plt.grid(True)
    plt.show()

def train_model(X, y, model, learning_rate=0.1, epochs=100):
    """训练模型"""
    # 将输入转换为Value对象
    X_value = [[Value(x) for x in sample] for sample in X]
    
    # 存储损失历史
    loss_history = []
    
    # 训练循环
    for epoch in range(epochs):
        # 前向传播
        ypred = [model(x) for x in X_value]
        
        # 计算损失（均方误差）
        loss = sum((yout - Value(ygt))**2 for ygt, yout in zip(y, ypred)) / len(y)
        loss_history.append(loss.data)
        
        # 反向传播
        # 清零梯度
        for p in model.parameters():
            p.grad = 0.0
        # 计算梯度
        loss.backward()
        
        # 更新参数（梯度下降）
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        # 打印损失
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.data:.4f}')
    
    return loss_history

def plot_decision_boundary(model, X, y):
    """可视化决策边界"""
    # 设置网格范围
    h = 0.1  # 网格步长
    x_min, x_max = min([x[0] for x in X]) - 1, max([x[0] for x in X]) + 1
    y_min, y_max = min([x[1] for x in X]) - 1, max([x[1] for x in X]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点的类别
    Z = []
    for i in range(len(xx.ravel())):
        # 接上一部分的代码
        x_val = xx.ravel()[i]
        y_val = yy.ravel()[i]
        pred = model([Value(x_val), Value(y_val)]).data
        Z.append(1 if pred > 0 else -1)
    Z = np.array(Z).reshape(xx.shape)
    
    # 绘制决策边界和数据点
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)
    plt.scatter([x[0] for i, x in enumerate(X) if y[i] > 0], 
                [x[1] for i, x in enumerate(X) if y[i] > 0], 
                c='r', marker='o', label='类别 1')
    plt.scatter([x[0] for i, x in enumerate(X) if y[i] < 0], 
                [x[1] for i, x in enumerate(X) if y[i] < 0], 
                c='b', marker='x', label='类别 2')
    plt.legend()
    plt.title('决策边界')
    plt.grid(True)
    plt.show()

def evaluate_model(model, X, y):
    """评估模型性能"""
    # 将输入转换为Value对象
    X_value = [[Value(x) for x in sample] for sample in X]
    
    # 计算准确率
    correct = 0
    for i, x in enumerate(X_value):
        pred = model(x).data
        pred_class = 1 if pred > 0 else -1
        if pred_class == y[i]:
            correct += 1
    
    accuracy = correct / len(y) * 100
    print(f"模型准确率: {accuracy:.2f}%")
    return accuracy

def main():
    """主函数"""
    print("Micrograd示例: 使用神经网络解决分类问题")
    
    # 生成数据集
    X, y = generate_dataset()
    
    # 可视化数据集
    print("数据集可视化:")
    plot_data(X, y)
    
    # 创建模型
    print("\n创建神经网络模型...")
    model = MLP(2, [16, 16, 1])
    
    # 训练模型
    print("\n开始训练模型...")
    loss_history = train_model(X, y, model, learning_rate=0.1, epochs=300)
    
    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.grid(True)
    plt.show()
    
    # 可视化决策边界
    print("\n决策边界可视化:")
    plot_decision_boundary(model, X, y)
    
    # 评估模型
    print("\n模型评估:")
    evaluate_model(model, X, y)
    
    print("\n示例完成!")

if __name__ == "__main__":
    main()