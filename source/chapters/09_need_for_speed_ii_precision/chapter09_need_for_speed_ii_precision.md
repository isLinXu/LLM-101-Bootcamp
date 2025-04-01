---
file_format: mystnb
kernelspec:
  name: python3
---
# 第9章：速度提升II：精度(Precision)

## 9.1 数值精度基础

在构建故事讲述AI大语言模型的过程中，数值精度是影响训练和推理速度的关键因素之一。合理选择和使用不同的数值精度可以显著提高计算效率，同时保持模型性能。本章我们将深入探讨数值精度的基础知识、混合精度训练的原理与实现，以及不同精度在故事生成中的应用。

### 9.1.1 浮点数表示

在计算机中，浮点数是表示实数的一种方式，它由符号位、指数和尾数组成。IEEE 754标准定义了几种常用的浮点数格式：

1. **单精度浮点数（FP32）**：
   - 32位表示：1位符号位，8位指数，23位尾数
   - 精度范围：约7位十进制数字
   - 数值范围：约±1.18 × 10^-38 到 ±3.4 × 10^38

2. **双精度浮点数（FP64）**：
   - 64位表示：1位符号位，11位指数，52位尾数
   - 精度范围：约16位十进制数字
   - 数值范围：约±2.23 × 10^-308 到 ±1.80 × 10^308

3. **半精度浮点数（FP16）**：
   - 16位表示：1位符号位，5位指数，10位尾数
   - 精度范围：约3-4位十进制数字
   - 数值范围：约±6.10 × 10^-5 到 ±65504

4. **脑浮点数（BF16）**：
   - 16位表示：1位符号位，8位指数，7位尾数
   - 与FP32相同的指数范围，但精度降低
   - 特别适合深度学习应用

5. **8位浮点数（FP8）**：
   - 8位表示：1位符号位，4位指数，3位尾数（E4M3格式）或其他变种
   - 极其有限的精度，但在某些应用中足够
   - 新兴的深度学习优化格式

下图展示了不同浮点格式的位分配：

```
FP32: Sign(1) | Exponent(8) | Mantissa(23)
FP64: Sign(1) | Exponent(11) | Mantissa(52)
FP16: Sign(1) | Exponent(5) | Mantissa(10)
BF16: Sign(1) | Exponent(8) | Mantissa(7)
FP8:  Sign(1) | Exponent(4) | Mantissa(3)
```

### 9.1.2 精度与计算效率的关系

数值精度直接影响计算效率和内存使用：

1. **计算速度**：
   - 较低精度（如FP16）的计算通常比高精度（如FP32）快2-8倍
   - 现代GPU的Tensor Cores专门优化了FP16和BF16计算
   - 8位精度可以进一步提高计算速度

2. **内存使用**：
   - FP16使用的内存是FP32的一半
   - 这意味着可以加载更大的模型或使用更大的批量大小
   - 减少内存传输也提高了整体性能

3. **能耗效率**：
   - 低精度计算通常能耗更低
   - 对于移动设备和边缘计算尤为重要

4. **硬件利用率**：
   - 低精度操作通常能更好地利用硬件资源
   - 例如，NVIDIA的Tensor Cores在FP16上性能最佳

### 9.1.3 精度与模型质量的权衡

然而，降低精度并非没有代价：

1. **数值范围限制**：
   - FP16的数值范围远小于FP32
   - 可能导致上溢（overflow）或下溢（underflow）

2. **舍入误差**：
   - 低精度表示会引入更多舍入误差
   - 误差可能在深层网络中累积

3. **训练稳定性**：
   - 纯FP16训练通常不稳定
   - 梯度更新和权重累积特别容易受到精度影响

4. **特殊操作敏感性**：
   - 某些操作（如归一化、指数、对数）对精度特别敏感
   - 可能需要在高精度下执行

因此，在实际应用中，我们需要在计算效率和模型质量之间找到平衡点。混合精度训练就是为解决这一问题而设计的。

## 9.2 混合精度训练原理

混合精度训练是一种在保持模型精度的同时提高训练速度和减少内存使用的技术。它的核心思想是在同一网络中使用不同的数值精度：在对精度不敏感的操作中使用低精度，在对精度敏感的操作中使用高精度。

### 9.2.1 混合精度训练的基本原则

混合精度训练基于以下几个关键原则：

1. **保持主权重副本**：
   - 在FP32中保存模型权重的主副本
   - 这确保了长期训练的数值稳定性

2. **前向和反向传播使用低精度**：
   - 将FP32权重转换为FP16进行前向传播
   - 在FP16中计算梯度
   - 这加速了计算密集型操作

3. **权重更新使用高精度**：
   - 将FP16梯度转换回FP32
   - 在FP32中进行权重更新
   - 这保持了更新的精确性

4. **损失缩放**：
   - 将损失值乘以一个缩放因子（通常是2的幂）
   - 这防止梯度在FP16表示中下溢
   - 在应用梯度前再除以相同的缩放因子

### 9.2.2 混合精度训练的工作流程

一个典型的混合精度训练循环如下：

1. **权重转换**：将FP32权重转换为FP16
2. **前向传播**：使用FP16权重和激活值进行前向传播
3. **损失缩放**：将损失乘以缩放因子
4. **反向传播**：计算FP16梯度
5. **梯度转换**：将FP16梯度转换回FP32，并除以缩放因子
6. **权重更新**：在FP32中更新主权重副本

这个过程可以用以下伪代码表示：

```python
# 初始化
model_fp32 = create_model()  # FP32主权重
optimizer = create_optimizer(model_fp32.parameters())
scaler = create_loss_scaler()  # 损失缩放器

# 训练循环
for inputs, targets in dataloader:
    # 将输入数据转换为FP16
    inputs_fp16 = inputs.to(dtype=torch.float16)
    
    # 前向传播（FP16）
    with autocast():  # 自动将操作转换为FP16
        outputs_fp16 = model_fp32(inputs_fp16)  # 内部使用FP16权重的副本
        loss = criterion(outputs_fp16, targets)
    
    # 损失缩放
    scaled_loss = scaler.scale(loss)
    
    # 反向传播（FP16梯度）
    scaled_loss.backward()
    
    # 梯度缩放和更新（FP32）
    scaler.step(optimizer)  # 内部将梯度转换为FP32并除以缩放因子
    
    # 更新损失缩放因子
    scaler.update()
    
    # 清零梯度
    optimizer.zero_grad()
```

### 9.2.3 损失缩放详解

损失缩放是混合精度训练中的关键技术，它解决了FP16梯度下溢的问题。

**为什么需要损失缩放？**

在深度学习中，梯度值通常很小，特别是在深层网络中。FP16的最小正规化数约为6 × 10^-5，这意味着任何小于此值的梯度都会被下溢为零。这会导致权重无法更新，训练停滞。

**损失缩放的工作原理：**

1. 将损失值乘以一个大的缩放因子（如2^8=256）
2. 由于反向传播的线性性质，所有梯度也会被同比例放大
3. 这使得小梯度值能够在FP16范围内表示
4. 在应用梯度前，将其除以相同的缩放因子，恢复原始尺度

**静态与动态损失缩放：**

1. **静态损失缩放**：
   - 使用固定的缩放因子
   - 简单但需要手动调整
   - 不同模型可能需要不同的最佳缩放因子

2. **动态损失缩放**：
   - 自动调整缩放因子
   - 当检测到梯度溢出时减小缩放因子
   - 一段时间内没有溢出时增大缩放因子
   - 更灵活，适应不同训练阶段

以下是动态损失缩放的简化实现：

```python
class DynamicLossScaler:
    def __init__(self, init_scale=2**15, scale_factor=2, scale_window=2000):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.iter_counter = 0
        self.last_overflow_iter = -1
    
    def scale_loss(self, loss):
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
    
    def update_scale(self, overflow):
        if overflow:
            # 梯度溢出，减小缩放因子
            self.scale = max(1, self.scale / self.scale_factor)
            self.last_overflow_iter = self.iter_counter
        elif (self.iter_counter - self.last_overflow_iter) >= self.scale_window:
            # 一段时间没有溢出，增大缩放因子
            self.scale *= self.scale_factor
            self.last_overflow_iter = self.iter_counter
        
        self.iter_counter += 1
    
    def check_overflow(self, params):
        # 检查梯度是否包含NaN或Inf
        for param in params:
            if param.grad is not None:
                grad = param.grad.data
                if not torch.isfinite(grad).all():
                    return True
        return False
```

## 9.3 实现混合精度训练

现在，让我们看看如何在实际中实现混合精度训练，特别是使用现代深度学习框架。

### 9.3.1 PyTorch中的混合精度训练

PyTorch提供了`torch.cuda.amp`（自动混合精度）模块，使混合精度训练变得简单。

**基本用法：**

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 创建模型和优化器
model = create_model().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 创建梯度缩放器
scaler = GradScaler()

# 训练循环
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 自动混合精度前向传播
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 缩放损失并反向传播
        scaler.scale(loss).backward()
        
        # 缩放优化器步骤
        scaler.step(optimizer)
        
        # 更新缩放因子
        scaler.update()
```

**`autocast`上下文管理器：**

`autocast`上下文管理器会自动将操作转换为适当的精度：
- 大多数操作使用FP16
- 对精度敏感的操作（如归一化）保持FP32
- 输入和输出类型会自动处理

**`GradScaler`类：**

`GradScaler`处理损失缩放的所有细节：
- 自动缩放损失
- 检测梯度溢出
- 动态调整缩放因子
- 在溢出时跳过优化器步骤

**自定义精度策略：**

对于更精细的控制，可以自定义哪些操作使用哪种精度：

```python
# 自定义精度策略
from torch.cuda.amp import autocast

# 默认使用FP16
with autocast():
    # 这些操作在FP16中执行
    hidden = model.encoder(inputs)
    
    # 对于特定操作，可以临时禁用autocast
    with autocast(enabled=False):
        # 这些操作在FP32中执行
        normalized = layer_norm(hidden)
    
    # 回到FP16
    output = model.decoder(normalized)
```

### 9.3.2 TensorFlow中的混合精度训练

TensorFlow也提供了混合精度训练的支持，通过`mixed_precision`模块。

**基本用法：**

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision

# 设置全局策略
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 创建模型
model = create_model()

# 编译模型（优化器会自动处理损失缩放）
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# 训练模型
model.fit(train_dataset, epochs=num_epochs)
```

**自定义训练循环：**

对于更精细的控制，可以使用自定义训练循环：

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision

# 设置策略
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 创建模型和优化器
model = create_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

# 训练循环
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        # 前向传播
        predictions = model(inputs, training=True)
        # 计算损失
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets, predictions)
        # 应用损失缩放
        scaled_loss = optimizer.get_scaled_loss(loss)
    
    # 计算缩放后的梯度
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # 取消梯度缩放
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    # 应用梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# 执行训练
for epoch in range(num_epochs):
    for inputs, targets in train_dataset:
        loss = train_step(inputs, targets)
```

### 9.3.3 混合精度训练的最佳实践

为了获得最佳的混合精度训练效果，以下是一些实践建议：

1. **使用支持Tensor Cores的GPU**：
   - NVIDIA Volta、Turing、Ampere或更新架构
   - 这些GPU在FP16操作上有硬件加速

2. **调整批量大小**：
   - 由于内存使用减少，可以增加批量大小
   - 更大的批量通常提高吞吐量和GPU利用率

3. **调整学习率**：
   - 更大的批量可能需要调整学习率
   - 考虑使用学习率缩放规则（如平方根缩放）

4. **监控损失缩放因子**：
   - 如果缩放因子频繁减小，可能表明数值不稳定
   - 可能需要调整模型架构或优化器参数

5. **注意精度敏感操作**：
   - 某些操作在低精度下可能导致问题
   - 考虑将这些操作保持在FP32中

6. **使用适当的归一化技术**：
   - 层归一化（Layer Normalization）在混合精度训练中通常比批量归一化更稳定
   - 考虑使用RMSNorm等变种

7. **检查溢出频率**：
   - 过于频繁的梯度溢出表明训练不稳定
   - 可能需要降低学习率或使用梯度裁剪

### 9.3.4 完整的混合精度训练示例

以下是一个完整的PyTorch混合精度训练示例，用于故事生成模型：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # 检查是否支持Tensor Cores
    compute_capability = torch.cuda.get_device_capability(0)
    supports_tensor_cores = compute_capability[0] >= 7
    print(f"Tensor Cores supported: {supports_tensor_cores}")
else:
    print("Using CPU. Mixed precision will not provide speedup.")

# 创建模型
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)
model = GPT2LMHeadModel(config)
model = model.to(device)

# 创建分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 准备数据集
def prepare_data(stories, tokenizer, max_length=512):
    inputs = []
    for story in stories:
        # 分词
        encodings = tokenizer(story, truncation=True, max_length=max_length, padding="max_length")
        inputs.append({
            "input_ids": torch.tensor(encodings["input_ids"]),
            "attention_mask": torch.tensor(encodings["attention_mask"])
        })
    return inputs

# 假设我们有一个故事数据集
stories = ["Once upon a time...", "In a galaxy far, far away..."]  # 实际应用中会有更多数据
train_dataset = prepare_data(stories, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 创建优化器
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# 创建学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# 创建梯度缩放器
scaler = GradScaler()

# 训练函数
def train_epoch(model, dataloader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # 将数据移至设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # 创建标签（偏移的输入）
        labels = input_ids.clone()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 混合精度前向传播
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        # 缩放损失并反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪（在取消缩放后）
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        
        # 更新学习率
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 训练循环
num_epochs = 3
for epoch in range(num_epochs):
    loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "storyteller_model.pt")

# 生成文本示例
def generate_text(model, tokenizer, prompt, max_length=100, device="cuda"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # 使用自动混合精度
        with autocast():
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 生成示例故事
prompt = "Once upon a time in a magical forest,"
generated_story = generate_text(model, tokenizer, prompt)
print(f"Generated story:\n{generated_story}")
```

## 9.4 精度与性能的权衡

不同的数值精度在性能和模型质量之间提供了不同的权衡点。了解这些权衡对于选择合适的精度策略至关重要。

### 9.4.1 不同精度格式的比较

让我们比较不同精度格式的特点：

| 精度格式 | 位宽 | 数值范围 | 精度 | 计算速度 | 内存使用 | 主要用途 |
|---------|-----|---------|------|---------|---------|---------|
| FP64 | 64位 | ±2.23×10^-308 到 ±1.80×10^308 | ~16位十进制 | 基准 | 基准×2 | 科学计算，高精度需求 |
| FP32 | 32位 | ±1.18×10^-38 到 ±3.4×10^38 | ~7位十进制 | 基准 | 基准 | 训练主权重，精度敏感操作 |
| FP16 | 16位 | ±6.10×10^-5 到 ±65504 | ~3-4位十进制 | 2-8×加速 | 1/2基准 | 前向/反向传播，Tensor Cores |
| BF16 | 16位 | 与FP32相同 | ~2-3位十进制 | 2-8×加速 | 1/2基准 | 训练，更好的数值稳定性 |
| FP8 | 8位 | 有限 | ~1-2位十进制 | 4-16×加速 | 1/4基准 | 推理，部分训练操作 |
| INT8 | 8位 | -128 到 127 | 整数 | 4-16×加速 | 1/4基准 | 量化推理 |

### 9.4.2 FP16 vs BF16

FP16和BF16都是16位格式，但它们有重要区别：

**FP16（IEEE 半精度）**：
- 1位符号，5位指数，10位尾数
- 更高的精度，但数值范围有限
- 容易出现溢出问题
- 在NVIDIA GPU上有更广泛的支持

**BF16（脑浮点数）**：
- 1位符号，8位指数，7位尾数
- 与FP32相同的指数范围，但精度降低
- 更少的溢出问题，训练更稳定
- 在Google TPU和较新的NVIDIA GPU上支持

**选择指南**：
- 如果数值稳定性是主要关注点，选择BF16
- 如果精度更重要且数值范围可控，选择FP16
- 如果硬件同时支持两种格式，BF16通常是训练的更好选择

以下是在PyTorch中使用BF16的示例：

```python
import torch
from torch.cuda.amp import autocast

# 检查BF16支持
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    print("BF16 is supported!")
    
    # 使用BF16自动混合精度
    with autocast(dtype=torch.bfloat16):
        output = model(input)
else:
    print("BF16 is not supported, falling back to FP16")
    
    # 使用FP16自动混合精度
    with autocast():
        output = model(input)
```

### 9.4.3 FP8简介

FP8是一种新兴的8位浮点格式，专为深度学习设计。它有几种变体，最常见的是：

**E4M3（4位指数，3位尾数）**：
- 适用于前向传播和权重
- 提供更大的动态范围

**E5M2（5位指数，2位尾数）**：
- 适用于梯度
- 提供更大的数值范围，但精度更低

FP8的主要优势：
- 比FP16/BF16进一步减少内存使用
- 提高计算速度和能效
- 在某些操作中可以达到与更高精度相当的模型质量

FP8的挑战：
- 需要特殊的缩放技术来保持精度
- 硬件支持仍在发展中（如NVIDIA Hopper架构）
- 可能需要更复杂的训练策略

### 9.4.4 精度选择策略

为不同的操作选择合适的精度是混合精度训练的核心。以下是一些常见的策略：

**权重存储**：
- 主副本：FP32
- 计算副本：FP16/BF16
- 量化模型：INT8/FP8

**前向传播**：
- 大多数操作：FP16/BF16
- 精度敏感操作（如LayerNorm）：FP32
- 推理时可考虑：INT8/FP8

**反向传播**：
- 梯度计算：FP16/BF16
- 梯度累积：FP32
- 需要损失缩放

**优化器状态和更新**：
- 动量、方差等状态：FP32
- 权重更新：FP32

**激活值**：
- 存储：FP16/BF16
- 检查点重计算：可能需要FP32

以下是一个实现这种精度选择策略的PyTorch示例：

```python
import torch
import torch.nn as nn

class MixedPrecisionModule(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # 主要计算在FP16/BF16中
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        
        # 转换为FP32进行精度敏感操作
        x_fp32 = x.to(torch.float32)
        residual_fp32 = residual.to(torch.float32)
        
        # 在FP32中执行残差连接和层归一化
        output_fp32 = self.layer_norm(x_fp32 + residual_fp32)
        
        # 返回到原始精度
        output = output_fp32.to(x.dtype)
        
        return output
```

## 9.5 精度问题排查与解决

在使用混合精度训练时，可能会遇到各种数值问题。了解如何识别和解决这些问题对于成功训练至关重要。

### 9.5.1 常见精度问题

以下是混合精度训练中最常见的问题：

1. **梯度下溢**：
   - 症状：权重停止更新，训练停滞
   - 原因：梯度值太小，在FP16中表示为零
   - 解决方案：增加损失缩放因子

2. **梯度爆炸**：
   - 症状：损失突然变为NaN或Inf
   - 原因：梯度值太大，超出FP16范围
   - 解决方案：梯度裁剪，减小学习率

3. **权重更新不稳定**：
   - 症状：训练不稳定，性能波动大
   - 原因：累积的舍入误差
   - 解决方案：在FP32中进行权重更新

4. **激活值溢出**：
   - 症状：前向传播中出现NaN
   - 原因：中间激活值超出FP16范围
   - 解决方案：检查并修改模型架构，使用更稳定的归一化

5. **损失缩放因子震荡**：
   - 症状：缩放因子频繁增加和减少
   - 原因：训练不稳定
   - 解决方案：调整优化器参数，使用更保守的缩放策略

### 9.5.2 调试工具和技术

以下工具和技术可以帮助识别和解决精度问题：

1. **梯度和激活值监控**：

```python
def check_tensor_values(tensor, name):
    """检查张量的统计信息"""
    if not torch.isfinite(tensor).all():
        print(f"{name} contains NaN or Inf")
    
    stats = {
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "std": tensor.std().item()
    }
    print(f"{name} stats: {stats}")

# 在训练循环中使用
for inputs, targets in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 检查前向传播值
    check_tensor_values(outputs, "outputs")
    check_tensor_values(loss, "loss")
    
    loss.backward()
    
    # 检查梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            check_tensor_values(param.grad, f"grad_{name}")
```

2. **梯度直方图**：

```python
from torch.utils.tensorboard import SummaryWriter

# 创建TensorBoard写入器
writer = SummaryWriter("logs/mixed_precision")

# 在训练循环中记录梯度直方图
def log_gradients(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, step)

# 在训练循环中使用
for step, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # 记录梯度
    log_gradients(model, step)
    
    optimizer.step()
```

3. **损失缩放因子监控**：

```python
# 使用PyTorch的GradScaler
scaler = GradScaler()

# 在训练循环中
for step, (inputs, targets) in enumerate(dataloader):
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    
    # 检查是否有梯度溢出
    overflow = scaler._check_inf_per_device(optimizer)[0]
    if overflow:
        print(f"Step {step}: Gradient overflow detected")
    
    # 更新参数
    scaler.step(optimizer)
    scaler.update()
    
    # 记录当前缩放因子
    current_scale = scaler.get_scale()
    print(f"Step {step}: Loss scale = {current_scale}")
```

4. **精度比较实验**：

```python
def train_with_precision(model, dataloader, precision="mixed"):
    """使用不同精度训练模型"""
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if precision == "mixed":
        # 混合精度训练
        scaler = GradScaler()
        use_autocast = True
    elif precision == "fp32":
        # 全FP32训练
        use_autocast = False
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    losses = []
    for inputs, targets in dataloader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        optimizer.zero_grad()
        
        if use_autocast:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
    
    return losses

# 比较不同精度
fp32_losses = train_with_precision(model.clone(), dataloader, precision="fp32")
mixed_losses = train_with_precision(model.clone(), dataloader, precision="mixed")

# 绘制损失比较
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(fp32_losses, label="FP32")
plt.plot(mixed_losses, label="Mixed Precision")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Comparison: FP32 vs Mixed Precision")
plt.savefig("precision_comparison.png")
```

### 9.5.3 解决方案和最佳实践

以下是解决混合精度训练中常见问题的策略：

1. **梯度下溢解决方案**：
   - 使用动态损失缩放
   - 增加初始损失缩放因子
   - 使用BF16代替FP16（如果硬件支持）

2. **梯度爆炸解决方案**：
   - 实施梯度裁剪（`torch.nn.utils.clip_grad_norm_`）
   - 减小学习率
   - 使用更稳定的优化器（如AdamW）

3. **数值稳定性改进**：
   - 使用层归一化代替批量归一化
   - 在精度敏感操作中使用FP32
   - 考虑使用残差缩放（Residual Scaling）

4. **初始化调整**：
   - 为混合精度训练调整权重初始化
   - 避免过大或过小的初始值

5. **架构修改**：
   - 使用对数值精度更稳健的激活函数（如GELU、SiLU）
   - 添加跳跃连接以改善梯度流
   - 考虑使用RMSNorm等更稳定的归一化变种

以下是一个综合示例，实现了多种精度问题解决方案：

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class NumericallyStableTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=12, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(50000, d_model)
        
        # 使用RMSNorm代替LayerNorm以提高数值稳定性
        self.norm_cls = RMSNorm
        
        # 创建Transformer层
        self.layers = nn.ModuleList([
            TransformerLayerWithPrecisionControl(d_model, nhead)
            for _ in range(num_layers)
        ])
        
        self.final_norm = self.norm_cls(d_model)
        self.lm_head = nn.Linear(d_model, 50000)
        
        # 应用特殊初始化
        self._init_weights()
    
    def _init_weights(self):
        """为混合精度训练优化的初始化"""
        for name, param in self.named_parameters():
            if "norm" in name and "weight" in name:
                # 归一化层权重初始化为1
                nn.init.ones_(param)
            elif "norm" in name and "bias" in name:
                # 归一化层偏置初始化为0
                nn.init.zeros_(param)
            elif "weight" in name:
                # 线性层权重使用较小的初始值
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits

class RMSNorm(nn.Module):
    """RMSNorm提供比LayerNorm更好的数值稳定性"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # 转换为FP32以提高精度
        x_fp32 = x.to(torch.float32)
        
        # 计算RMS
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # 应用缩放
        x_normalized = x_fp32 * rms * self.scale
        
        # 返回到原始精度
        return x_normalized.to(x.dtype)

class TransformerLayerWithPrecisionControl(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),  # 使用GELU而非ReLU以提高稳定性
            nn.Linear(4 * d_model, d_model)
        )
        
        # 残差缩放因子（帮助控制梯度尺度）
        self.res_scale = 1.0 / (2.0 * num_layers) ** 0.5
    
    def forward(self, x):
        # 自注意力块
        residual = x
        
        # 在FP32中进行归一化
        x_norm = self.norm1(x)
        
        # 自注意力（在原始精度中）
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        
        # 应用残差缩放并添加残差
        x = residual + attn_out * self.res_scale
        
        # 前馈块
        residual = x
        
        # 在FP32中进行归一化
        x_norm = self.norm2(x)
        
        # 前馈网络（在原始精度中）
        mlp_out = self.mlp(x_norm)
        
        # 应用残差缩放并添加残差
        x = residual + mlp_out * self.res_scale
        
        return x

# 训练函数
def train_with_robust_mixed_precision(model, dataloader, num_epochs=3):
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # 创建梯度缩放器（使用较大的初始缩放因子）
    scaler = GradScaler(init_scale=2**16)
    
    # 训练循环
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪（在取消缩放后）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 检查梯度是否包含NaN
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        print(f"NaN gradient detected in {name}")
                        valid_gradients = False
                        break
            
            # 只在梯度有效时更新参数
            if valid_gradients:
                scaler.step(optimizer)
            
            # 更新缩放因子
            scaler.update()
            
            # 打印当前缩放因子
            if random.random() < 0.01:  # 随机采样以减少输出
                print(f"Epoch {epoch}, Loss Scale: {scaler.get_scale()}")
```

## 9.6 不同精度在故事生成中的影响

数值精度不仅影响训练速度和效率，还可能影响生成故事的质量和特性。在本节中，我们将探讨不同精度对故事生成的影响。

### 9.6.1 精度对生成质量的影响

不同的数值精度可能以多种方式影响生成的故事：

1. **词汇多样性**：
   - 较低精度可能导致概率分布更加"尖锐"
   - 这可能减少生成文本的多样性
   - 在极端情况下，可能导致重复或套路化内容

2. **连贯性和流畅度**：
   - 精度不足可能影响模型捕捉微妙语言模式的能力
   - 这可能导致生成的故事中出现不自然的转折或不连贯的段落

3. **创意和独特性**：
   - 精度限制可能影响模型探索不太可能的词序列的能力
   - 这可能减少生成内容的创意性和独特性

4. **长文本一致性**：
   - 在生成长故事时，精度问题可能累积
   - 这可能导致故事后期出现主题漂移或情节不一致

以下是一个比较不同精度对故事生成影响的实验设计：

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_story_with_precision(model, tokenizer, prompt, precision, max_length=200):
    """使用指定精度生成故事"""
    model = model.cuda().eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    
    # 设置精度
    if precision == "fp32":
        dtype = torch.float32
    elif precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    # 转换模型权重
    model = model.to(dtype)
    
    # 生成文本
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成的文本
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

# 定义提示
prompt = "Once upon a time in a magical forest, a young fairy discovered a mysterious glowing stone. As she picked it up,"

# 使用不同精度生成故事
precisions = ["fp32", "fp16", "bf16"] if torch.cuda.is_bf16_supported() else ["fp32", "fp16"]
stories = {}

for precision in precisions:
    stories[precision] = generate_story_with_precision(model, tokenizer, prompt, precision)
    print(f"\n--- Story generated with {precision} ---\n")
    print(stories[precision])

# 分析生成的故事
def analyze_story(story):
    """简单分析生成的故事"""
    words = story.split()
    unique_words = set(words)
    
    analysis = {
        "length": len(words),
        "unique_words": len(unique_words),
        "lexical_diversity": len(unique_words) / len(words) if words else 0,
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
    
    return analysis

# 比较不同精度生成的故事
for precision, story in stories.items():
    analysis = analyze_story(story)
    print(f"\n--- Analysis for {precision} ---")
    for key, value in analysis.items():
        print(f"{key}: {value}")
```

### 9.6.2 精度对推理性能的影响

在故事生成的推理阶段，精度选择对性能有显著影响：

1. **延迟（Latency）**：
   - 较低精度通常可以减少生成每个词元的时间
   - 这对于交互式故事生成应用尤为重要

2. **吞吐量（Throughput）**：
   - 较低精度允许批处理更多请求
   - 这对于服务多用户的应用很有价值

3. **内存使用**：
   - 较低精度减少了模型的内存占用
   - 这允许在相同硬件上加载更大的模型

4. **能耗**：
   - 较低精度操作通常能耗更低
   - 这对于移动设备和边缘计算很重要

以下是一个比较不同精度对故事生成推理性能影响的基准测试：

```python
import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def benchmark_generation(model, tokenizer, prompt, precision, num_tokens=100, num_runs=10):
    """测量指定精度下的生成性能"""
    model = model.cuda().eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    
    # 设置精度
    if precision == "fp32":
        dtype = torch.float32
    elif precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    # 转换模型权重
    model = model.to(dtype)
    
    # 预热
    with torch.no_grad():
        model.generate(input_ids, max_length=input_ids.shape[1] + 10, num_return_sequences=1)
    
    # 测量生成时间
    latencies = []
    torch.cuda.synchronize()
    
    for _ in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            model.generate(
                input_ids,
                max_length=input_ids.shape[1] + num_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        latencies.append(end_time - start_time)
    
    # 计算性能指标
    avg_latency = sum(latencies) / len(latencies)
    tokens_per_second = num_tokens / avg_latency
    
    # 测量内存使用
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model.generate(
            input_ids,
            max_length=input_ids.shape[1] + num_tokens,
            do_sample=True,
            num_return_sequences=1
        )
    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    return {
        "precision": precision,
        "avg_latency": avg_latency,
        "tokens_per_second": tokens_per_second,
        "memory_usage_mb": memory_usage
    }

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

# 定义提示
prompt = "Once upon a time"

# 测试不同精度
precisions = ["fp32", "fp16", "bf16"] if torch.cuda.is_bf16_supported() else ["fp32", "fp16"]
results = []

for precision in precisions:
    result = benchmark_generation(model, tokenizer, prompt, precision)
    results.append(result)
    print(f"\n--- Performance with {precision} ---")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

# 绘制性能比较图
import matplotlib.pyplot as plt
import numpy as np

# 延迟比较
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
latencies = [result["avg_latency"] for result in results]
plt.bar(precisions, latencies)
plt.title("Average Latency (seconds)")
plt.ylabel("Seconds")
plt.grid(axis="y", alpha=0.3)

# 吞吐量比较
plt.subplot(1, 2, 2)
throughputs = [result["tokens_per_second"] for result in results]
plt.bar(precisions, throughputs)
plt.title("Throughput (tokens/second)")
plt.ylabel("Tokens per Second")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("precision_performance_comparison.png")
```

### 9.6.3 为故事生成选择最佳精度

基于前面的讨论，以下是为故事生成模型选择最佳精度的指南：

**训练阶段**：
- **大型模型（>1B参数）**：
  - 使用混合精度训练（FP16或BF16）
  - 如果训练不稳定，优先选择BF16（如果硬件支持）
  - 使用动态损失缩放和梯度裁剪

- **中型模型（100M-1B参数）**：
  - 使用混合精度训练
  - 监控训练稳定性，必要时调整损失缩放策略

- **小型模型（<100M参数）**：
  - 如果训练速度是主要关注点，使用混合精度
  - 如果训练稳定性更重要，可以考虑纯FP32训练

**推理阶段**：
- **在线交互式生成**：
  - 优先考虑低延迟
  - 使用FP16或INT8量化（如果质量可接受）
  - 考虑KV缓存优化（将在后续章节讨论）

- **批量故事生成**：
  - 优先考虑高吞吐量
  - 使用FP16或BF16
  - 优化批处理大小以最大化设备利用率

- **移动设备或边缘部署**：
  - 使用INT8量化或更激进的压缩技术
  - 考虑模型蒸馏以减小模型大小

**质量敏感的应用**：
- 对于需要高质量、创意性强的故事生成：
  - 在推理时使用FP32或BF16
  - 特别是对于长故事生成，高精度可以减少质量下降

- 对于特定风格或主题的故事：
  - 测试不同精度对风格保持的影响
  - 可能需要在某些层使用更高精度

以下是一个为故事生成实现精度自适应的示例：

```python
import torch
import time

class AdaptivePrecisionStoryGenerator:
    """根据需求自适应选择精度的故事生成器"""
    
    def __init__(self, model, tokenizer):
        self.model = model.cuda().eval()
        self.tokenizer = tokenizer
        self.available_precisions = ["fp32", "fp16"]
        if torch.cuda.is_bf16_supported():
            self.available_precisions.append("bf16")
    
    def _convert_model_precision(self, precision):
        """转换模型精度"""
        if precision == "fp32":
            self.model = self.model.to(torch.float32)
        elif precision == "fp16":
            self.model = self.model.to(torch.float16)
        elif precision == "bf16":
            self.model = self.model.to(torch.bfloat16)
        else:
            raise ValueError(f"Unsupported precision: {precision}")
    
    def generate_story(self, prompt, max_length=200, mode="balanced"):
        """生成故事，根据模式选择精度"""
        if mode == "quality":
            # 质量优先，使用FP32
            precision = "fp32"
        elif mode == "speed":
            # 速度优先，使用最低可用精度
            precision = "fp16" if "fp16" in self.available_precisions else "fp32"
        elif mode == "balanced":
            # 平衡模式，优先使用BF16
            precision = "bf16" if "bf16" in self.available_precisions else "fp16"
        elif mode == "adaptive":
            # 自适应模式，根据故事长度选择精度
            if max_length > 500:
                # 长故事使用更高精度以保持一致性
                precision = "fp32"
            elif max_length > 200:
                # 中等长度使用平衡精度
                precision = "bf16" if "bf16" in self.available_precisions else "fp16"
            else:
                # 短故事使用最低精度以获得速度
                precision = "fp16" if "fp16" in self.available_precisions else "fp32"
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        # 转换模型精度
        self._convert_model_precision(precision)
        
        # 编码提示
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        
        # 生成文本
        start_time = time.time()
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        
        # 解码生成的文本
        story = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return {
            "story": story,
            "precision_used": precision,
            "generation_time": generation_time,
            "tokens_per_second": max_length / generation_time
        }

# 使用示例
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
generator = AdaptivePrecisionStoryGenerator(model, tokenizer)

# 生成不同模式的故事
prompt = "The ancient dragon awoke from its thousand-year slumber. Its eyes glowed with"

modes = ["quality", "speed", "balanced", "adaptive"]
for mode in modes:
    result = generator.generate_story(prompt, max_length=150, mode=mode)
    print(f"\n--- Story generated in {mode} mode (using {result['precision_used']}) ---")
    print(f"Generation time: {result['generation_time']:.2f} seconds")
    print(f"Speed: {result['tokens_per_second']:.2f} tokens/second")
    print("\nStory:")
    print(result["story"])
```

## 9.7 总结与展望

在本章中，我们深入探讨了数值精度在大语言模型训练和推理中的重要性。我们介绍了不同的浮点格式（FP32、FP16、BF16、FP8），讨论了混合精度训练的原理和实现，分析了精度问题的排查与解决方法，并探讨了不同精度在故事生成中的影响。

混合精度训练是现代大语言模型训练的标准做法，它在保持模型质量的同时显著提高了训练速度和内存效率。通过合理选择不同操作的精度，并使用损失缩放等技术，我们可以克服低精度带来的挑战，充分利用现代硬件的计算能力。

随着大语言模型规模的不断增长，精度优化将变得越来越重要。未来的发展趋势包括：

1. **更低位宽格式**：FP8等更低位宽的浮点格式将变得更加普及，进一步提高计算效率。

2. **精度感知架构**：未来的模型架构可能会从设计上考虑数值精度，使模型在低精度下更加稳定。

3. **自适应精度**：训练和推理系统可能会动态调整不同操作的精度，根据实时需求优化性能和质量。

4. **硬件协同设计**：软件和硬件将更紧密地协同设计，以支持新的精度格式和混合精度计算模式。

5. **量化感知训练**：将量化考虑直接纳入训练过程，使模型在低精度下表现更好。

在下一章中，我们将探讨速度提升的另一个关键方面：分布式优化。我们将讨论如何在多个设备和多台机器上高效训练大型语言模型，包括数据并行、模型并行、流水线并行等技术，以及ZeRO等优化器。这些技术将使我们能够训练更大、更强大的故事生成模型。

**练习与思考**

1. 比较FP32、FP16和BF16在训练小型故事生成模型（如GPT-2 Small）时的性能和质量差异。
2. 实现一个自定义的损失缩放器，并比较静态损失缩放和动态损失缩放的效果。
3. 设计一个实验，测量不同精度对生成故事多样性和创意性的影响。
4. 探索如何使用PyTorch的Profiler工具分析混合精度训练中的性能瓶颈。
5. 实现一个精度自适应的推理系统，根据输入长度和计算资源动态选择最佳精度。

**参考资料**

1. Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., ... & Wu, H. (2018). Mixed Precision Training. In International Conference on Learning Representations.
2. NVIDIA. (2022). NVIDIA Automatic Mixed Precision for Deep Learning.
3. Kalamkar, D., Mudigere, D., Mellempudi, N., Das, D., Banerjee, K., Avancha, S., ... & Dubey, P. (2019). A Study of BFLOAT16 for Deep Learning Training. arXiv preprint arXiv:1905.12322.
4. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). 8-bit Optimizers via Block-wise Quantization. In International Conference on Learning Representations.
5. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems.
6. Kuchaiev, O., Ginsburg, B., Gitman, I., Lavrukhin, V., Li, J., Nguyen, H., ... & Micikevicius, P. (2018). Mixed-Precision Training for NLP and Speech Recognition with OpenSeq2Seq. arXiv preprint arXiv:1805.10387.
7. Narang, S., Diamos, G., Elsen, E., Micikevicius, P., Alben, J., Vainbrand, D., ... & Zhou, Y. (2018). Mixed Precision Training of Convolutional Neural Networks using Integer Operations. In International Conference on Learning Representations.
8. Sun, X., Choi, J., Chen, C. Y., Wang, N., Venkataramani, S., Srinivasan, V., ... & Gopalakrishnan, K. (2019). Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks. In Advances in Neural Information Processing Systems.
