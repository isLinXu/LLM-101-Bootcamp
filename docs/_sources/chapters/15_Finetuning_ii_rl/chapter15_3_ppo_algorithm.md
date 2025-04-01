---
file_format: mystnb
kernelspec:
  name: python3
---
# 第15章：强化学习微调 II: RL-15.3 近端策略优化(PPO)算法

## 15.3 近端策略优化(PPO)算法

在前两节中，我们探讨了强化学习的基础概念以及人类反馈的强化学习（RLHF）框架。RLHF为我们提供了一种将人类偏好整合到语言模型训练中的方法，但要实际实现这一目标，我们需要一个稳定且高效的强化学习算法。近端策略优化（Proximal Policy Optimization，PPO）正是RLHF中最常用的算法，它因其稳定性、样本效率和实现相对简单而受到广泛采用。本节将深入探讨PPO算法的数学基础、在语言模型中的适应性修改以及实现细节。

### PPO算法的数学基础

PPO算法是一种策略梯度方法，它通过迭代优化策略来最大化预期累积奖励。与传统的策略梯度方法相比，PPO引入了一种约束机制，限制每次更新中策略的变化幅度，从而提高训练的稳定性。

#### 策略梯度回顾

在深入PPO之前，让我们简要回顾策略梯度方法的基本原理。策略梯度方法直接优化策略函数 $\pi_\theta(a|s)$，其中 $\theta$ 是策略的参数。基本的策略梯度目标是最大化预期回报：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

其中 $\tau$ 是一个轨迹（状态-动作序列），$R(\tau)$ 是该轨迹的累积奖励。

策略梯度定理给出了目标函数的梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t]$$

其中 $R_t$ 是从时间步 $t$ 开始的折扣累积奖励。

#### 重要性采样与替代目标

PPO的一个关键创新是使用重要性采样来重用数据。假设我们有一个旧策略 $\pi_{\theta_{old}}$ 和一个新策略 $\pi_\theta$，我们可以使用重要性权重来调整期望：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{old}}}[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \cdot R(\tau)]$$

定义重要性权重比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，PPO提出了一个替代目标函数：

$$L^{CPI}(\theta) = \mathbb{E}_t[r_t(\theta) \cdot A_t]$$

其中 $A_t$ 是优势函数，估计动作 $a_t$ 相对于平均水平的好坏程度。

#### PPO的裁剪目标

PPO的核心创新是引入了一个裁剪机制，限制策略更新的幅度。裁剪后的目标函数为：

$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta) \cdot A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t)]$$

其中 $\epsilon$ 是一个小常数（通常为0.1或0.2），用于控制裁剪范围。这个目标函数有两个关键特性：

1. 当优势 $A_t$ 为正时，它鼓励增加动作的概率，但不超过 $(1+\epsilon)$ 倍的旧概率
2. 当优势 $A_t$ 为负时，它鼓励减少动作的概率，但不低于 $(1-\epsilon)$ 倍的旧概率

这种裁剪机制防止了过大的策略更新，提高了训练的稳定性。

#### 完整的PPO目标

实际应用中，PPO通常还包括熵奖励和值函数损失，完整的目标函数为：

$$L^{TOTAL}(\theta) = \mathbb{E}_t[L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)]$$

其中：
- $L^{VF}$ 是值函数损失，通常是均方误差：$(V_\theta(s_t) - V_t^{target})^2$
- $S$ 是策略的熵，鼓励探索
- $c_1$ 和 $c_2$ 是权重系数

### PPO在LLM中的适应性修改

将PPO应用于大型语言模型（LLMs）需要一些特殊的适应性修改，以处理语言生成的独特挑战。

#### KL散度约束

在语言模型中，保持生成文本的流畅性和连贯性至关重要。为此，PPO-RLHF通常添加一个额外的KL散度约束，限制优化后的策略与初始SFT模型之间的差异：

$$L^{RLHF}(\theta) = \mathbb{E}_t[L^{CLIP}(\theta) - \beta \cdot \text{KL}[\pi_\theta || \pi_{SFT}]]$$

其中 $\beta$ 是KL散度的权重系数，$\pi_{SFT}$ 是经过监督微调的初始模型。

这个约束有两个重要作用：
1. 防止模型"忘记"如何生成流畅的文本
2. 避免模型过度优化奖励，生成不自然或欺骗性的内容

#### 参考策略选择

在PPO-RLHF中，有三种常见的参考策略选择：

1. **SFT模型作为参考**：
   - 使用监督微调后的模型作为KL散度约束的参考
   - 优点：保持语言质量，防止偏离人类示范
   - 缺点：可能限制模型改进空间

2. **初始策略作为参考**：
   - 使用每次PPO更新前的策略作为参考
   - 优点：允许模型逐渐偏离SFT，更大的优化空间
   - 缺点：可能随时间累积偏差，逐渐偏离自然语言

3. **混合参考**：
   - 同时使用SFT模型和初始策略作为参考
   - 例如：$\text{KL}[\pi_\theta || \text{mix}(\pi_{SFT}, \pi_{\theta_{old}})]$
   - 平衡了稳定性和优化空间

#### 值函数设计

在语言模型中，值函数的设计也需要特殊考虑：

1. **共享vs分离架构**：
   - 共享架构：策略和值函数共享相同的语言模型主干
   - 分离架构：使用单独的模型作为值函数
   - 实践中，通常使用共享主干但分离头部的方法

2. **上下文表示**：
   - 值函数需要基于完整的上下文预测预期奖励
   - 常用方法是使用最后一个标记的隐藏状态，或对所有标记的隐藏状态进行池化

3. **奖励归因**：
   - 语言生成是一个序列决策过程，需要将最终奖励归因到各个决策步骤
   - 可以使用简单的折扣分配或更复杂的归因方法

### 实现细节与技巧

实现PPO-RLHF需要注意许多细节和技巧，以确保训练的稳定性和效率。

#### 轨迹收集

在语言模型中，轨迹收集涉及生成完整的文本序列：

```python
def collect_trajectories(policy_model, prompts, reward_model, tokenizer, device, 
                         generation_kwargs):
    """收集轨迹（生成的文本序列及其奖励）"""
    trajectories = []
    
    for prompt in prompts:
        # 编码提示
        prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 生成响应
        with torch.no_grad():
            output_ids = policy_model.generate(
                prompt_tokens.input_ids,
                **generation_kwargs
            )
        
        # 分离提示和响应
        response_ids = output_ids[0][prompt_tokens.input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # 计算奖励
        full_text = prompt + response_text
        reward_input = tokenizer(full_text, return_tensors="pt").to(device)
        with torch.no_grad():
            reward = reward_model(**reward_input).logits[0].item()
        
        # 计算每个标记的对数概率
        log_probs = []
        with torch.no_grad():
            for i in range(len(response_ids)):
                # 获取到当前位置的输入
                input_ids = torch.cat([prompt_tokens.input_ids[0], response_ids[:i+1]])
                input_ids = input_ids.unsqueeze(0)
                
                # 前向传播
                outputs = policy_model(input_ids)
                logits = outputs.logits[0, -1, :]  # 获取最后一个标记的logits
                
                # 计算对数概率
                log_prob = F.log_softmax(logits, dim=-1)[response_ids[i]].item()
                log_probs.append(log_prob)
        
        # 构建轨迹
        trajectory = {
            "prompt": prompt,
            "response": response_text,
            "response_ids": response_ids.tolist(),
            "log_probs": log_probs,
            "reward": reward
        }
        
        trajectories.append(trajectory)
    
    return trajectories
```

#### 优势估计

优势函数估计动作相对于平均水平的好坏程度。在语言模型中，可以使用广义优势估计（GAE）：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """计算广义优势估计（GAE）"""
    advantages = []
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # 最后一步的优势
            delta = rewards[t] - values[t]
        else:
            # 中间步骤的优势
            delta = rewards[t] + gamma * values[t+1] - values[t]
        
        # 计算GAE
        advantage = delta + gamma * lam * last_advantage
        advantages.insert(0, advantage)
        last_advantage = advantage
    
    return advantages
```

#### 批处理和优化

PPO通常使用小批量更新，并进行多个优化周期：

```python
def optimize_policy(policy_model, value_model, trajectories, optimizer, 
                    old_policy_model, sft_model, epochs=4, batch_size=64, 
                    clip_epsilon=0.2, kl_coef=0.1, value_coef=0.5, entropy_coef=0.01):
    """使用PPO优化策略"""
    # 准备数据
    all_response_ids = []
    all_log_probs = []
    all_rewards = []
    all_advantages = []
    all_returns = []
    all_values = []
    
    for traj in trajectories:
        all_response_ids.extend(traj["response_ids"])
        all_log_probs.extend(traj["log_probs"])
        
        # 计算每个标记的奖励（简化版，实际应使用更复杂的归因）
        sequence_length = len(traj["log_probs"])
        per_token_rewards = [traj["reward"] / sequence_length] * sequence_length
        all_rewards.extend(per_token_rewards)
        
        # 计算值函数预测
        with torch.no_grad():
            values = value_model(torch.tensor(traj["response_ids"]).unsqueeze(0)).values[0].tolist()
        all_values.extend(values)
    
    # 计算优势和回报
    advantages = compute_advantages(all_rewards, all_values)
    returns = [adv + val for adv, val in zip(advantages, all_values)]
    
    all_advantages.extend(advantages)
    all_returns.extend(returns)
    
    # 转换为张量
    response_ids = torch.tensor(all_response_ids)
    old_log_probs = torch.tensor(all_log_probs)
    advantages = torch.tensor(all_advantages)
    returns = torch.tensor(all_returns)
    
    # 多个优化周期
    for _ in range(epochs):
        # 创建数据加载器
        dataset = TensorDataset(response_ids, old_log_probs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for batch in dataloader:
            batch_response_ids, batch_old_log_probs, batch_advantages, batch_returns = batch
            
            # 计算新策略的对数概率
            outputs = policy_model(batch_response_ids.unsqueeze(1))
            logits = outputs.logits[:, 0, :]
            log_probs = F.log_softmax(logits, dim=-1)
            batch_new_log_probs = torch.gather(log_probs, 1, batch_response_ids.unsqueeze(1)).squeeze(1)
            
            # 计算重要性权重比率
            ratios = torch.exp(batch_new_log_probs - batch_old_log_probs)
            
            # 计算裁剪后的目标
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算值函数损失
            value_preds = value_model(batch_response_ids.unsqueeze(1)).values.squeeze(1)
            value_loss = F.mse_loss(value_preds, batch_returns)
            
            # 计算熵奖励
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
            
            # 计算KL散度（与SFT模型）
            with torch.no_grad():
                sft_outputs = sft_model(batch_response_ids.unsqueeze(1))
                sft_logits = sft_outputs.logits[:, 0, :]
                sft_log_probs = F.log_softmax(sft_logits, dim=-1)
            
            kl_div = F.kl_div(log_probs, torch.exp(sft_log_probs), reduction='batchmean')
            
            # 总损失
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + kl_coef * kl_div
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### 实现稳定性技巧

实现PPO-RLHF时，以下技巧有助于提高训练稳定性：

1. **梯度裁剪**：
   - 限制梯度范数，防止梯度爆炸
   - 通常设置为1.0或0.5

2. **学习率调度**：
   - 使用线性或余弦学习率衰减
   - 从较小的学习率开始（如1e-5）

3. **归一化优势**：
   - 在每个批次内归一化优势值
   - 减少训练方差，提高稳定性

4. **值函数早停**：
   - 监控值函数损失，防止过拟合
   - 可以使用单独的学习率和早停标准

5. **KL散度自适应调整**：
   - 动态调整KL散度系数
   - 如果KL散度过大，增加系数；如果过小，减少系数

6. **混合精度训练**：
   - 使用FP16或BF16减少内存需求
   - 特别适用于大型模型

### 实现细节与优化

在实际应用中，PPO-RLHF的实现需要考虑许多工程细节，以确保训练的效率和稳定性。

#### 分布式训练

对于大型语言模型，分布式训练通常是必要的：

```python
def setup_distributed_training(model, world_size):
    """设置分布式训练"""
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # 包装模型进行分布式训练
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )
    
    return model, local_rank
```

#### 内存优化

PPO训练可能非常内存密集，特别是对于大型模型：

1. **梯度检查点（Gradient Checkpointing）**：
   - 以计算时间换取内存
   - 在前向传播中只保存关键激活，其他激活在反向传播时重新计算

2. **参数高效微调**：
   - 使用LoRA等技术减少可训练参数
   - 显著降低内存需求，同时保持性能

3. **混合精度训练**：
   - 使用FP16或BF16进行计算
   - 减少内存使用并加速训练

4. **优化批处理**：
   - 使用梯度累积增加有效批量大小
   - 动态调整批量大小以最大化GPU利用率

#### 完整的PPO-RLHF实现

以下是一个更完整的PPO-RLHF实现示例，结合了上述技术和优化：

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import wandb

class PPOTrainer:
    def __init__(
        self,
        policy_model,
        ref_model,
        reward_model,
        tokenizer,
        optimizer,
        device,
        config
    ):
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # 创建值函数头
        self.value_head = nn.Linear(
            policy_model.config.hidden_size, 1
        ).to(device)
        self.value_optimizer = torch.optim.Adam(
            self.value_head.parameters(), lr=config.value_lr
        )
        
        # 设置学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps
        )
        
        # 初始化KL控制器
        self.kl_ctl = AdaptiveKLController(
            init_kl_coef=config.init_kl_coef,
            target=config.target_kl,
            horizon=config.kl_horizon
        )
        
        # 启用梯度检查点以节省内存
        if config.gradient_checkpointing:
            self.policy_model.gradient_checkpointing_enable()
    
    def get_value_predictions(self, input_ids, attention_mask):
        """获取值函数预测"""
        with torch.no_grad():
            outputs = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]
            values = self.value_head(last_hidden).squeeze(-1)
            
            # 只保留响应部分的值
            response_mask = attention_mask.clone()
            response_mask[:, :input_ids.shape[1] - attention_mask.sum(dim=1)] = 0
            masked_values = values * response_mask
            
            return masked_values
    
    def compute_rewards(self, sequences):
        """计算奖励"""
        with torch.no_grad():
            rewards = self.reward_model(sequences).logits
        return rewards
    
    def generate_responses(self, prompts, generation_kwargs):
        """生成响应并收集轨迹"""
        trajectories = []
        
        for prompt in tqdm(prompts, desc="Generating responses"):
            # 编码提示
            prompt_tokens = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            
            # 生成响应
            with torch.no_grad():
                output_ids = self.policy_model.generate(
                    prompt_tokens.input_ids,
                    attention_mask=prompt_tokens.attention_mask,
                    **generation_kwargs
                )
            
            # 分离提示和响应
            prompt_length = prompt_tokens.input_ids.shape[1]
            response_ids = output_ids[0][prompt_length:]
            
            # 解码响应
            response_text = self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            )
            
            # 构建完整序列
            full_ids = torch.cat([prompt_tokens.input_ids[0], response_ids])
            full_mask = torch.ones_like(full_ids)
            
            # 计算奖励
            reward = self.compute_rewards(
                full_ids.unsqueeze(0), full_mask.unsqueeze(0)
            )[0].item()
            
            # 计算值函数预测
            values = self.get_value_predictions(
                full_ids.unsqueeze(0), full_mask.unsqueeze(0)
            )[0][prompt_length:].tolist()
            
            # 计算参考模型的对数概率
            ref_logprobs = self.compute_logprobs(
                self.ref_model, full_ids.unsqueeze(0), response_ids
            )[0].tolist()
            
            # 计算策略模型的对数概率
            policy_logprobs = self.compute_logprobs(
                self.policy_model, full_ids.unsqueeze(0), response_ids
            )[0].tolist()
            
            # 构建轨迹
            trajectory = {
                "prompt": prompt,
                "response": response_text,
                "response_ids": response_ids.tolist(),
                "values": values,
                "reward": reward,
                "ref_logprobs": ref_logprobs,
                "policy_logprobs": policy_logprobs
            }
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def compute_logprobs(self, model, input_ids, target_ids):
        """计算目标标记的对数概率"""
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
            # 获取目标标记的对数概率
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = torch.gather(
                log_probs[:, :-1], 2, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            return target_log_probs
    
    def compute_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        """计算广义优势估计（GAE）"""
        advantages = []
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 最后一步的优势
                delta = rewards[t] - values[t]
            else:
                # 中间步骤的优势
                delta = rewards[t] + gamma * values[t+1] - values[t]
            
            # 计算GAE
            advantage = delta + gamma * lam * last_advantage
            advantages.insert(0, advantage)
            last_advantage = advantage
   <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>