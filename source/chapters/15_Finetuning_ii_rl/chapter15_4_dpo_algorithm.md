---
file_format: mystnb
kernelspec:
  name: python3
---
# 第15章：强化学习微调 II: RL-## 15.4 直接偏好优化(DPO)算法

## 15.4 直接偏好优化(DPO)算法

在前面的章节中，我们详细探讨了强化学习的基础概念、人类反馈的强化学习（RLHF）框架以及近端策略优化（PPO）算法。虽然PPO-RLHF已经成为优化大型语言模型的主流方法，但它的实现复杂且计算资源需求高，这限制了其在资源受限环境中的应用。为了解决这些挑战，研究人员开发了直接偏好优化（Direct Preference Optimization，DPO）算法，它提供了一种更简单、更高效的方法来将人类偏好整合到语言模型训练中。本节将深入探讨DPO的原理、实现和应用，特别是在故事讲述AI的背景下。

### DPO的基本原理

DPO的核心思想是直接从人类偏好数据中学习最优策略，而无需显式的奖励建模和强化学习过程。这种方法基于一个关键洞察：在RLHF中，奖励模型和PPO优化可以被一个统一的目标函数所替代，该函数直接从偏好数据中学习。

#### 从RLHF到DPO的理论推导

为了理解DPO的原理，我们需要回顾RLHF的数学基础，并看看如何将其简化。

在RLHF中，我们首先训练一个奖励模型 $r_\phi(x, y)$，然后使用强化学习（通常是PPO）来优化策略 $\pi_\theta(y|x)$。这个过程可以表示为以下优化问题：

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)}[r_\phi(x, y)] - \beta \cdot \text{KL}[\pi_\theta(y|x) || \pi_{\text{ref}}(y|x)]$$

其中 $\beta$ 是KL散度的权重系数，$\pi_{\text{ref}}$ 是参考策略（通常是SFT模型）。

DPO的关键洞察是，在最优条件下，策略 $\pi_\theta$ 和奖励函数 $r_\phi$ 之间存在一个明确的关系：

$$\pi_\theta(y|x) \propto \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{1}{\beta}r_\phi(x, y)\right)$$

这个关系表明，最优策略是参考策略的指数加权版本，权重由奖励函数决定。

通过数学推导，我们可以将这个关系重写为：

$$r_\phi(x, y_w) - r_\phi(x, y_l) = \beta \cdot \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \cdot \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$

其中 $y_w$ 和 $y_l$ 分别是人类偏好的"获胜"和"失败"响应。

这个等式的关键意义在于，我们可以直接从偏好数据中学习策略 $\pi_\theta$，而无需显式训练奖励模型 $r_\phi$。这就是DPO的核心思想。

#### DPO的目标函数

基于上述推导，DPO提出了以下目标函数：

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log\sigma\left(\beta \cdot \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \cdot \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

其中 $\sigma$ 是sigmoid函数，$\mathcal{D}$ 是人类偏好数据集。

这个目标函数可以解释为：我们希望最大化模型在人类偏好数据上正确预测偏好的概率。具体来说，如果人类偏好 $y_w$ 而不是 $y_l$，那么我们希望模型对 $y_w$ 的相对偏好（相对于参考模型）高于对 $y_l$ 的相对偏好。

### DPO与PPO-RLHF的比较

DPO相比传统的PPO-RLHF有几个显著优势：

1. **简化的训练流程**：
   - DPO不需要单独训练奖励模型
   - 不需要复杂的强化学习优化
   - 整个训练过程类似于标准的监督学习

2. **计算效率**：
   - DPO通常比PPO-RLHF快10-100倍
   - 内存需求显著降低
   - 可以在单个GPU上训练较大的模型

3. **稳定性**：
   - 避免了强化学习中的不稳定性
   - 不需要调整复杂的RL超参数
   - 训练过程更加稳定可靠

4. **理论保证**：
   - DPO与RLHF在理论上是等价的
   - 在理想条件下，两者应该收敛到相同的策略

然而，DPO也有一些潜在的局限性：

1. **灵活性**：
   - PPO-RLHF可以更灵活地调整奖励函数
   - 在复杂任务中，显式的奖励建模可能提供更多控制

2. **探索能力**：
   - DPO缺乏强化学习中的探索机制
   - 可能在某些需要创新性解决方案的任务中表现不佳

3. **多步决策**：
   - 在需要长期规划的任务中，PPO可能有优势
   - DPO主要针对单步决策优化

### DPO的实现细节

实现DPO相对简单，主要包括数据准备、模型训练和超参数调整三个方面。

#### 数据准备

与RLHF类似，DPO需要人类偏好数据，通常是(提示, 获胜响应, 失败响应)的三元组：

```python
def prepare_dpo_dataset(preference_data, tokenizer, max_length=512):
    """准备DPO训练数据集"""
    dataset = []
    
    for item in preference_data:
        prompt = item["prompt"]
        chosen = item["chosen"]  # 人类偏好的响应
        rejected = item["rejected"]  # 人类不偏好的响应
        
        # 编码提示和响应
        prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        chosen_tokens = tokenizer(chosen, return_tensors="pt", truncation=True, max_length=max_length)
        rejected_tokens = tokenizer(rejected, return_tensors="pt", truncation=True, max_length=max_length)
        
        dataset.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_tokens": prompt_tokens,
            "chosen_tokens": chosen_tokens,
            "rejected_tokens": rejected_tokens
        })
    
    return dataset
```

#### DPO训练循环

DPO的训练循环相对简单，类似于标准的监督学习：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class DPOTrainer:
    def __init__(
        self,
        policy_model,
        reference_model,
        tokenizer,
        beta=0.1,
        device="cuda",
        learning_rate=1e-5
    ):
        self.policy_model = policy_model.to(device)
        self.reference_model = reference_model.to(device)
        self.tokenizer = tokenizer
        self.beta = beta
        self.device = device
        
        # 冻结参考模型
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # 设置优化器
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=learning_rate
        )
    
    def get_logps(self, model, prompt_tokens, response_tokens):
        """计算模型对响应的对数概率"""
        # 将提示和响应连接起来
        input_ids = torch.cat([prompt_tokens.input_ids, response_tokens.input_ids], dim=1)
        attention_mask = torch.cat([prompt_tokens.attention_mask, response_tokens.attention_mask], dim=1)
        
        # 前向传播
        with torch.set_grad_enabled(model is self.policy_model):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 只考虑响应部分的logits
            response_logits = logits[:, prompt_tokens.input_ids.shape[1]-1:-1, :]
            
            # 获取目标标记
            response_targets = response_tokens.input_ids
            
            # 计算对数概率
            log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, 2, response_targets.unsqueeze(-1)
            ).squeeze(-1)
            
            # 计算序列对数概率（忽略padding）
            response_mask = response_tokens.attention_mask
            sequence_log_probs = (token_log_probs * response_mask).sum(dim=1) / response_mask.sum(dim=1)
            
            return sequence_log_probs
    
    def compute_dpo_loss(self, prompt_tokens, chosen_tokens, rejected_tokens):
        """计算DPO损失"""
        # 获取策略模型的对数概率
        policy_chosen_logps = self.get_logps(self.policy_model, prompt_tokens, chosen_tokens)
        policy_rejected_logps = self.get_logps(self.policy_model, prompt_tokens, rejected_tokens)
        
        # 获取参考模型的对数概率
        with torch.no_grad():
            reference_chosen_logps = self.get_logps(self.reference_model, prompt_tokens, chosen_tokens)
            reference_rejected_logps = self.get_logps(self.reference_model, prompt_tokens, rejected_tokens)
        
        # 计算对数比率
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        
        # 计算DPO损失
        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        # 计算准确率（预测正确的偏好比例）
        accuracy = (logits > 0).float().mean()
        
        return loss, accuracy, chosen_logratios.mean(), rejected_logratios.mean()
    
    def train_step(self, batch):
        """执行一个DPO训练步骤"""
        # 将数据移到设备
        prompt_tokens = {k: v.to(self.device) for k, v in batch["prompt_tokens"].items()}
        chosen_tokens = {k: v.to(self.device) for k, v in batch["chosen_tokens"].items()}
        rejected_tokens = {k: v.to(self.device) for k, v in batch["rejected_tokens"].items()}
        
        # 计算损失
        loss, accuracy, chosen_logratio, rejected_logratio = self.compute_dpo_loss(
            prompt_tokens, chosen_tokens, rejected_tokens
        )
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_logratio": chosen_logratio.item(),
            "rejected_logratio": rejected_logratio.item()
        }
    
    def train(self, train_dataset, batch_size=4, epochs=3):
        """执行完整的DPO训练循环"""
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 训练循环
        for epoch in range(epochs):
            epoch_stats = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                stats = self.train_step(batch)
                epoch_stats.append(stats)
            
            # 计算平均统计信息
            avg_stats = {
                k: sum(s[k] for s in epoch_stats) / len(epoch_stats)
                for k in epoch_stats[0].keys()
            }
            
            print(f"Epoch {epoch+1} stats: {avg_stats}")
        
        return self.policy_model

class DPODataset(Dataset):
    """DPO训练数据集"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

#### 超参数调整

DPO的主要超参数是 $\beta$，它控制KL散度的权重。较小的 $\beta$ 允许模型更大程度地偏离参考模型，而较大的 $\beta$ 则限制模型的变化。

一般来说，$\beta$ 的取值范围在0.1到0.5之间，具体取值需要根据任务和数据集进行调整。以下是一些调整建议：

1. **初始值**：从0.1开始，这是一个相对保守的值
2. **增大 $\beta$**：如果模型偏离参考模型太多，导致语言质量下降
3. **减小 $\beta$**：如果模型变化太小，无法有效学习人类偏好

除了 $\beta$ 外，其他重要的超参数包括：

1. **学习率**：通常在1e-6到5e-5之间，比标准微调略小
2. **批量大小**：尽可能大，受限于GPU内存
3. **训练轮次**：通常1-3轮足够，过多可能导致过拟合

### DPO在故事生成中的应用

DPO特别适合故事生成任务，因为故事质量的评估高度主观，难以通过显式规则捕捉。以下是DPO在故事讲述AI中的几个具体应用：

#### 风格优化

DPO可以用来优化故事的风格，使其符合特定的文学风格或目标受众：

```python
# 收集风格偏好数据
style_preference_data = [
    {
        "prompt": "写一个关于宇航员的故事",
        "chosen": "星辰大海的召唤，让李明选择了宇航员这条艰难的道路...",  # 成人风格
        "rejected": "小明想成为宇航员，他每天努力学习，希望有一天能飞向太空..."  # 儿童风格
    },
    # 更多风格偏好对...
]

# 使用DPO优化风格
style_model = train_dpo(base_model, style_preference_data, beta=0.2)
```

#### 情节复杂度调整

DPO可以用来调整故事的情节复杂度，使其适合不同年龄段的读者：

```python
# 收集情节复杂度偏好数据
complexity_preference_data = [
    {
        "prompt": "写一个关于寻宝的冒险故事",
        "chosen": "藏宝图上的线索指向了三个不同的地点，每个地点都隐藏着解开最终宝藏位置的关键...",  # 复杂情节
        "rejected": "小明找到了一张藏宝图，他按照地图找到了宝藏，非常开心。"  # 简单情节
    },
    # 更多复杂度偏好对...
]

# 使用DPO优化情节复杂度
complexity_model = train_dpo(base_model, complexity_preference_data, beta=0.15)
```

#### 教育价值增强

对于儿童故事，DPO可以用来增强故事的教育价值：

```python
# 收集教育价值偏好数据
educational_preference_data = [
    {
        "prompt": "写一个关于友谊的故事",
        "chosen": "小明和小红一开始总是争吵，但通过一次共同解决问题的经历，他们学会了相互理解和尊重...",  # 有教育价值
        "rejected": "小明和小红是好朋友，他们每天一起玩耍，非常开心。"  # 缺乏教育价值
    },
    # 更多教育价值偏好对...
]

# 使用DPO优化教育价值
educational_model = train_dpo(base_model, educational_preference_data, beta=0.1)
```

#### 多维度优化

在实际应用中，我们通常需要同时优化故事的多个方面。这可以通过组合不同类型的偏好数据来实现：

```python
# 组合多维度偏好数据
combined_preference_data = style_preference_data + complexity_preference_data + educational_preference_data

# 使用DPO进行多维度优化
combined_model = train_dpo(base_model, combined_preference_data, beta=0.2)
```

或者，可以使用多阶段训练，先优化一个方面，再优化另一个方面：

```python
# 第一阶段：优化风格
style_model = train_dpo(base_model, style_preference_data, beta=0.2)

# 第二阶段：在风格模型基础上优化情节复杂度
complexity_model = train_dpo(style_model, complexity_preference_data, beta=0.15)

# 第三阶段：在复杂度模型基础上优化教育价值
final_model = train_dpo(complexity_model, educational_preference_data, beta=0.1)
```

### DPO的高级技术和扩展

随着研究的深入，DPO已经发展出了一些高级技术和扩展，进一步提升了其性能和适用性。

#### 对比DPO (Contrastive DPO)

对比DPO是DPO的一个变体，它通过对比学习的方式优化模型。具体来说，它不仅考虑偏好对之间的差异，还考虑不同提示下的响应之间的差异：

```python
def compute_contrastive_dpo_loss(self, batch):
    """计算对比DPO损失"""
    # 标准DPO损失
    dpo_loss, accuracy, _, _ = self.compute_dpo_loss(
        batch["prompt_tokens"],
        batch["chosen_tokens"],
        batch["rejected_tokens"]
    )
    
    # 对比损失
    contrastive_loss = 0.0
    batch_size = len(batch["prompt"])
    
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                # 计算不同提示下的对比损失
                cross_prompt_loss, _, _, _ = self.compute_dpo_loss(
                    batch["prompt_tokens"][i:i+1],
                    batch["chosen_tokens"][i:i+1],
                    batch["chosen_tokens"][j:j+1]
                )
                contrastive_loss += cross_prompt_loss
    
    contrastive_loss /= batch_size * (batch_size - 1)
    
    # 组合损失
    total_loss = dpo_loss + self.contrastive_weight * contrastive_loss
    
    return total_loss, accuracy
```

对比DPO在某些任务上表现优于标准DPO，特别是在数据有限的情况下。

#### 迭代DPO (Iterative DPO)

迭代DPO是一种迭代优化策略，它通过多轮DPO训练逐步改进模型：

1. 使用初始模型生成响应
2. 收集这些响应的人类偏好
3. 使用DPO优化模型
4. 使用优化后的模型生成新的响应
5. 重复步骤2-4

这种方法可以逐步提升模型性能，特别是在初始模型质量不高的情况下。

#### 多目标DPO (Multi-objective DPO)

多目标DPO旨在同时优化多个可能相互冲突的目标，如创意性和连贯性：

```python
def compute_multi_objective_dpo_loss(self, batch):
    """计算多目标DPO损失"""
    # 计算不同目标的DPO损失
    coherence_loss, _, _, _ = self.compute_dpo_loss(
        batch["prompt_tokens"],
        batch["coherence_chosen_tokens"],
        batch["coherence_rejected_tokens"]
    )
    
    creativity_loss, _, _, _ = self.compute_dpo_loss(
        batch["prompt_tokens"],
        batch["creativity_chosen_tokens"],
        batch["creativity_rejected_tokens"]
    )
    
    engagement_loss, _, _, _ = self.compute_dpo_loss(
        batch["prompt_tokens"],
        batch["engagement_chosen_tokens"],
        batch["engagement_rejected_tokens"]
    )
    
    # 加权组合
    total_loss = (
        self.coherence_weight * coherence_loss +
        self.creativity_weight * creativity_loss +
        self.engagement_weight * engagement_loss
    )
    
    return total_loss
```

多目标DPO可以在不同目标之间取得平衡，生成更全面优化的故事。

### DPO的实际案例：儿童故事优化

为了具体说明DPO在故事讲述AI中的应用，让我们考虑一个实际案例：优化一个儿童故事生成器。

#### 背景和目标

我们有一个通过SFT初步训练的儿童故事生成模型，但发现它存在以下问题：
- 有时使用过于复杂的语言和概念
- 故事结构不够清晰，缺乏明确的开始、中间和结束
- 教育信息传递不够有效

我们的目标是使用DPO优化模型，使其生成：
- 语言简单明了，适合5-8岁儿童
- 结构清晰，有明确的故事弧
- 包含积极的教育信息，但不生硬

#### 数据收集

首先，我们收集人类偏好数据：

1. 生成多个故事对：
   - 使用SFT模型为100个不同的提示生成多个故事版本
   - 每个提示生成3-5个不同的版本，使用不同的采样参数

2. 收集人类评估：
   - 招募儿童教育专家、家长和儿童文学作家作为评估者
   - 让他们比较同一提示下的不同故事版本
   - 记录他们的偏好和评价理由

3. 构建偏好数据集：
   - 从评估结果中提取偏好对（获胜版本和失败版本）
   - 确保数据集覆盖不同类型的故事和主题
   - 最终收集约500个偏好对

#### DPO实现

接下来，我们使用DPO优化模型：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 加载模型和分词器
sft_model = AutoModelForCausalLM.from_pretrained("children-story-sft-model")
tokenizer = AutoTokenizer.from_pretrained("children-story-sft-model")

# 创建参考模型（复制SFT模型）
ref_model = AutoModelForCausalLM.from_pretrained("children-story-sft-model")

# 加载偏好数据
preference_data = load_dataset("json", data_files="children_story_preferences.jsonl")

# 准备DPO数据集
def preprocess_function(examples):
    # 编码提示和响应
    prompt_tokens = tokenizer(examples["prompt"], return_tensors="pt", padding=True, truncation=True)
    chosen_tokens = tokenizer(examples["chosen"], return_tensors="pt", padding=True, truncation=True)
    rejected_tokens = tokenizer(examples["rejected"], return_tensors="pt", padding=True, truncation=True)
    
    return {
        "prompt": examples["prompt"],
        "chosen": examples["chosen"],
        "rejected": examples["rejected"],
        "prompt_tokens": prompt_tokens,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens
    }

processed_data = preference_data.map(preprocess_function, batched=True)

# 创建DPO训练器
dpo_trainer = DPOTrainer(
    policy_model=sft_model,
    reference_model=ref_model,
    tokenizer=tokenizer,
    beta=0.2,  # 适中的KL权重
    learning_rate=2e-5
)

# 训练模型
optimized_model = dpo_trainer.train(
    processed_data["train"],
    batch_size=4,
    epochs=3
)

# 保存优化后的模型
optimized_model.save_pretrained("children-story-dpo-model")
tokenizer.save_pretrained("children-story-dpo-model")
```

#### 评估和结果

训练完成后，我们对优化后的模型进行全面评估：

1. **自动评估**：
   - 计算语言复杂度指标（如Flesch-Kincaid可读性分数）
   - 分析故事结构（如引言、冲突、解决方案的存在）
   - 检测教育元素的存在

2. **人工评估**：
   - 让评估者比较SFT模型和DPO模型生成的故事
   - 收集关于语言适当性、结构清晰度和教育价值的反馈
   - 进行盲测，评估者不知道哪个故事来自哪个模型

3. **目标受众测试**：
   - 让5-8岁的儿童听取或阅读生成的故事
   - 观察他们的参与度和理解程度
   - 收集他们的喜好和反馈

评估结果显示，DPO优化后的模型在以下方面取得了显著改进：

1. **语言适当性**：
   - 可读性分数降低，更适合目标年龄段
   - 句子长度减少，词汇更简单
   - 减少了抽象概念和复杂表达

2. **故事结构**：
   - 93%的故事有明确的开始、中间和结束（相比SFT模型的78%）
   - 角色和情节更加清晰
   - 故事弧更加完整和连贯

3. **教育价值**：
   - 85%的故事包含明确的教育信息（相比SFT模型的62%）
   - 教育信息更加自然地融入故事
   - 积极信息的传递更加有效

4. **整体质量**：
   - 在盲测中，评估者在72%的情况下偏好DPO模型生成的故事
   - 儿童对DPO模型生成的故事表现出更高的参与度和理解度
   - 家长和教育工作者对DPO模型的评价更高

这个案例展示了DPO在优化故事讲述AI方面的有效性，特别是在需要平衡多个目标（如语言适当性、结构清晰度和教育价值）的情况下。

### DPO的局限性和未来发展

虽然DPO提供了一种简单高效的方法来优化语言模型，但它仍然存在一些局限性：

1. **数据依赖**：
   - DPO严重依赖高质量的人类偏好数据
   - 收集这样的数据可能成本高昂且耗时
   - 数据中的偏见可能被模型放大

2. **长期规划**：
   - DPO主要针对单步决策优化
   - 在需<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>