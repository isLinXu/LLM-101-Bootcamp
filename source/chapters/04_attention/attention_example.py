import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from attention_utils import scaled_dot_product_attention, MultiHeadAttention, plot_attention_weights

# 创建一个简单的序列复制任务
def generate_copy_task(batch_size, seq_len, vocab_size):
    """生成一个复制任务的数据集：模型需要复制输入序列"""
    # 生成随机输入序列
    input_seq = torch.randint(1, vocab_size, (batch_size, seq_len))
    # 目标是复制输入序列
    target_seq = input_seq.clone()
    return input_seq, target_seq

# 创建一个使用注意力机制的简单Seq2Seq模型
class AttentionCopyModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads):
        super(AttentionCopyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.decoder = nn.LSTM(d_model * 2, d_model, batch_first=True)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        batch_size, src_len = src.shape
        
        # 嵌入输入序列
        src_embedded = self.embedding(src)
        
        # 编码
        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)
        
        # 如果没有提供目标，我们只预测一个步骤（用于推理）
        if tgt is None:
            tgt_len = src_len
        else:
            tgt_len = tgt.shape[1]
        
        # 准备输出张量
        outputs = torch.zeros(batch_size, tgt_len, self.output_layer.out_features)
        
        # 初始化解码器输入（起始符号）
        decoder_input = torch.zeros(batch_size, 1, d_model).to(src.device)
        
        # 保存所有注意力权重以便可视化
        all_attention_weights = torch.zeros(batch_size, tgt_len, src_len)
        
        for t in range(tgt_len):
            # 计算当前输入的注意力
            context, attention_weights = self.attention(
                hidden.transpose(0, 1), encoder_outputs, encoder_outputs)
            
            # 保存注意力权重
            all_attention_weights[:, t, :] = attention_weights[:, 0, 0, :]
            
            # 连接注意力上下文和当前输入
            rnn_input = torch.cat([decoder_input, context], dim=2)
            
            # 解码
            decoder_output, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))
            
            # 生成输出词汇分布
            prediction = self.output_layer(decoder_output.squeeze(1))
            outputs[:, t] = prediction
            
            # 计算下一时间步的输入
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force and tgt is not None:
                next_input = self.embedding(tgt[:, t]).unsqueeze(1)
            else:
                next_input = self.embedding(prediction.argmax(dim=1)).unsqueeze(1)
            
            decoder_input = next_input
            
        return outputs, all_attention_weights

# 训练函数
def train_model(model, data_loader, epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (src, tgt) in enumerate(data_loader):
            optimizer.zero_grad()
            
            # 前向传播
            output, _ = model(src, tgt)
            
            # 计算损失
            output_flat = output.view(-1, output.shape[-1])
            tgt_flat = tgt.view(-1)
            loss = criterion(output_flat, tgt_flat)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
    
    return losses

# 主函数
if __name__ == "__main__":
    # 任务参数
    batch_size = 64
    seq_len = 10
    vocab_size = 100
    d_model = 64
    num_heads = 4
    epochs = 5
    
    # 生成数据
    src, tgt = generate_copy_task(batch_size, seq_len, vocab_size)
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(src, tgt)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建模型
    model = AttentionCopyModel(vocab_size, d_model, num_heads)
    
    # 训练模型
    losses = train_model(model, data_loader, epochs=epochs)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('attention_training_loss.png')
    
    # 测试模型
    test_src, test_tgt = generate_copy_task(1, seq_len, vocab_size)
    output, attention_weights = model(test_src)
    
    # 显示预测结果
    print("输入序列:", test_src[0].numpy())
    print("预测序列:", output[0].argmax(dim=1).numpy())
    print("目标序列:", test_tgt[0].numpy())
    
    # 可视化注意力权重
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[0].numpy(), cmap='viridis')
    plt.xlabel('源序列位置')
    plt.ylabel('目标序列位置')
    plt.colorbar(label='注意力权重')
    plt.title('注意力权重可视化')
    plt.savefig('attention_weights.png') 