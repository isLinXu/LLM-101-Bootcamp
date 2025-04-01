---
file_format: mystnb
kernelspec:
  name: python3
---
# 第15章：强化学习微调 II: RL-15.1 强化学习基础

## 15.1 强化学习基础

在第14章中，我们探讨了如何通过监督式微调（SFT）使大型语言模型适应故事讲述任务。虽然SFT能够显著提升模型的能力，但它仍然存在一些固有的局限性。特别是，SFT仅能让模型学习模仿训练数据中的模式，而无法直接优化模型以满足人类的真实偏好或特定的质量标准。这就是强化学习（Reinforcement Learning，RL）微调发挥作用的地方。

强化学习微调允许我们根据特定的奖励信号来调整模型行为，使其生成的内容更符合我们的期望。在故事讲述AI的背景下，这意味着我们可以训练模型生成更有创意、更连贯、更符合特定风格或价值观的故事。本节将介绍强化学习的核心概念，为后续章节中的高级技术奠定基础。

### 强化学习的核心概念

强化学习是机器学习的一个分支，它关注如何使智能体（agent）在与环境交互的过程中，通过试错学习来最大化累积奖励。与监督学习不同，强化学习不依赖于标记数据，而是通过奖励信号来指导学习过程。

在强化学习框架中，有几个关键概念：

1. **智能体（Agent）**：做出决策的实体，在我们的情境中，这是故事讲述AI模型。

2. **环境（Environment）**：智能体交互的外部系统，对于语言模型来说，这可以是模拟的对话环境或评估系统。

3. **状态（State）**：环境的当前情况，对于故事生成任务，这可能是当前的故事上下文或提示。

4. **动作（Action）**：智能体可以执行的操作，在语言生成中，这是模型生成的下一个词或句子。

5. **奖励（Reward）**：环境对智能体动作的反馈，用于评估动作的好坏，例如故事质量的评分。

6. **策略（Policy）**：智能体的决策规则，决定在给定状态下应该采取什么动作。对于语言模型，这是决定下一个词的概率分布。

7. **价值函数（Value Function）**：估计在给定状态下，预期未来奖励的函数，帮助智能体评估不同状态的价值。

8. **轨迹（Trajectory）**：一系列状态、动作和奖励的序列，代表智能体与环境交互的一次完整过程。

强化学习的目标是找到一个最优策略，使智能体能够获得最大的累积奖励。在故事讲述AI的背景下，这意味着找到一种生成策略，使模型能够创作出获得最高人类评价的故事。

### 强化学习在NLP中的应用挑战

虽然强化学习在游戏、机器人控制等领域取得了显著成功，但将其应用于自然语言处理（NLP）和大型语言模型面临一些独特的挑战：

1. **高维离散动作空间**：
   - 语言模型的动作空间是词汇表的大小，通常包含数万到数十万个词元
   - 这使得传统的强化学习算法难以有效探索所有可能的动作
   - 需要特殊的技术来处理如此大的动作空间

2. **稀疏奖励问题**：
   - 在故事生成中，有意义的奖励通常只在完整故事生成后才能获得
   - 中间步骤（单个词或句子）的贡献难以评估
   - 这导致了信用分配问题：很难确定哪些特定决策导致了最终的好或坏结果

3. **奖励设计的复杂性**：
   - 故事质量是多维度的，包括创意性、连贯性、吸引力等
   - 这些维度难以量化，且可能相互冲突
   - 设计能够捕捉所有这些方面的奖励函数非常困难

4. **样本效率低**：
   - 传统强化学习算法需要大量样本才能收敛
   - 在语言模型中，生成和评估每个样本的成本很高
   - 这使得纯粹的试错学习在实践中不可行

5. **不稳定性**：
   - 语言生成的随机性使得训练过程更加不稳定
   - 小的参数变化可能导致生成质量的显著波动
   - 需要特殊的稳定化技术

6. **人类偏好的主观性**：
   - 不同人对故事质量的评价可能有很大差异
   - 偏好可能随时间和上下文变化
   - 这使得构建一致的奖励信号变得困难

为了应对这些挑战，研究人员开发了一系列专门针对语言模型的强化学习方法，其中最著名的是人类反馈的强化学习（RLHF），我们将在下一节详细讨论。

### 奖励函数设计

奖励函数是强化学习的核心组成部分，它定义了我们希望模型优化的目标。在故事讲述AI的背景下，设计一个有效的奖励函数需要考虑多个方面：

#### 奖励函数的类型

1. **基于规则的奖励**：
   - 使用预定义的规则和启发式方法评估故事质量
   - 例如，可以奖励词汇多样性、句子结构变化、适当的故事长度等
   - 优点：实现简单，不需要额外训练
   - 缺点：难以捕捉故事的高级特性，如创意性或情感影响

2. **基于模型的奖励**：
   - 训练专门的奖励模型来评估故事质量
   - 这些模型通常基于人类偏好数据进行训练
   - 优点：能够学习复杂的质量标准，更接近人类判断
   - 缺点：需要大量标记数据，可能继承数据中的偏见

3. **混合奖励**：
   - 结合规则和模型的优势
   - 例如，使用规则检查基本质量标准，然后使用模型评估更高级的特性
   - 优点：更全面的评估，可以平衡不同方面
   - 缺点：设计复杂，需要仔细调整不同组件的权重

#### 故事质量的评估维度

设计奖励函数时，需要考虑故事质量的多个维度：

1. **连贯性（Coherence）**：
   - 故事的逻辑流程是否合理
   - 角色、情节和设定是否一致
   - 可以通过检测矛盾或跟踪实体关系来评估

2. **创意性（Creativity）**：
   - 故事是否包含原创元素
   - 是否避免陈词滥调和过度使用常见模式
   - 可以通过与现有故事的相似度或惊奇度来衡量

3. **吸引力（Engagement）**：
   - 故事是否能够吸引读者注意力
   - 是否有情感共鸣和紧张感
   - 可以通过预测读者反应或使用代理指标（如阅读时间）来评估

4. **风格一致性（Stylistic Consistency）**：
   - 故事是否保持一致的语言风格
   - 是否符合特定的文体要求（如童话、科幻等）
   - 可以通过风格分类器或与参考文本的相似度来衡量

5. **价值观对齐（Value Alignment）**：
   - 故事是否符合预期的道德标准和价值观
   - 是否避免有害或不适当的内容
   - 可以通过内容安全过滤器或专门的对齐模型来评估

6. **教育价值（Educational Value）**：
   - 对于儿童故事，是否包含有教育意义的元素
   - 是否传递积极的信息
   - 可以通过检测特定主题或道德教训来评估

#### 奖励函数实现示例

以下是一个简化的奖励函数实现示例，它结合了多个维度来评估故事质量：

```python
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy

class StoryRewardFunction:
    def __init__(self):
        # 加载预训练的连贯性评估模型
        self.coherence_model = AutoModelForSequenceClassification.from_pretrained("coherence-model")
        self.coherence_tokenizer = AutoTokenizer.from_pretrained("coherence-model")
        
        # 加载预训练的创意性评估模型
        self.creativity_model = AutoModelForSequenceClassification.from_pretrained("creativity-model")
        self.creativity_tokenizer = AutoTokenizer.from_pretrained("creativity-model")
        
        # 加载预训练的安全性评估模型
        self.safety_model = AutoModelForSequenceClassification.from_pretrained("safety-model")
        self.safety_tokenizer = AutoTokenizer.from_pretrained("safety-model")
        
        # 加载spaCy用于语言分析
        self.nlp = spacy.load("zh_core_web_lg")
        
        # 设置各维度的权重
        self.weights = {
            "coherence": 0.3,
            "creativity": 0.3,
            "engagement": 0.2,
            "safety": 0.2
        }
    
    def evaluate_coherence(self, story):
        """评估故事的连贯性"""
        inputs = self.coherence_tokenizer(story, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.coherence_model(**inputs)
        
        # 假设模型输出连贯性分数（0-1）
        coherence_score = outputs.logits[0][1].item()
        return coherence_score
    
    def evaluate_creativity(self, story):
        """评估故事的创意性"""
        inputs = self.creativity_tokenizer(story, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.creativity_model(**inputs)
        
        # 假设模型输出创意性分数（0-1）
        creativity_score = outputs.logits[0][1].item()
        return creativity_score
    
    def evaluate_engagement(self, story):
        """评估故事的吸引力（使用启发式方法）"""
        doc = self.nlp(story)
        
        # 计算情感词的比例
        emotion_words = 0
        total_words = 0
        for token in doc:
            if token.is_alpha and not token.is_stop:
                total_words += 1
                # 简化的情感检测，实际应使用情感词典
                if token.pos_ in ["ADJ", "ADV"] and token.has_vector:
                    emotion_words += 1
        
        emotion_ratio = emotion_words / max(total_words, 1)
        
        # 计算对话的比例（作为互动性的代理指标）
        dialogue_sentences = 0
        total_sentences = len(list(doc.sents))
        for sent in doc.sents:
            if '"' in sent.text or '"' in sent.text or '：' in sent.text:
                dialogue_sentences += 1
        
        dialogue_ratio = dialogue_sentences / max(total_sentences, 1)
        
        # 计算句子长度变化（作为节奏变化的代理指标）
        sentence_lengths = [len(sent) for sent in doc.sents]
        length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        normalized_variance = min(length_variance / 100, 1)  # 归一化到0-1
        
        # 组合指标
        engagement_score = 0.4 * emotion_ratio + 0.3 * dialogue_ratio + 0.3 * normalized_variance
        return engagement_score
    
    def evaluate_safety(self, story):
        """评估故事的安全性和价值观对齐"""
        inputs = self.safety_tokenizer(story, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.safety_model(**inputs)
        
        # 假设模型输出安全性分数（0-1，越高越安全）
        safety_score = outputs.logits[0][1].item()
        return safety_score
    
    def calculate_reward(self, story):
        """计算故事的总体奖励分数"""
        # 评估各个维度
        coherence = self.evaluate_coherence(story)
        creativity = self.evaluate_creativity(story)
        engagement = self.evaluate_engagement(story)
        safety = self.evaluate_safety(story)
        
        # 计算加权总分
        total_reward = (
            self.weights["coherence"] * coherence +
            self.weights["creativity"] * creativity +
            self.weights["engagement"] * engagement +
            self.weights["safety"] * safety
        )
        
        # 返回总分和各维度分数
        return {
            "total_reward": total_reward,
            "coherence": coherence,
            "creativity": creativity,
            "engagement": engagement,
            "safety": safety
        }
```

这个示例展示了如何结合基于模型的评估和启发式方法来创建一个多维度的奖励函数。在实际应用中，你可能需要根据特定需求调整各个维度的权重，并使用更复杂的评估模型。

### 策略与价值函数

在强化学习中，策略和价值函数是两个核心概念，它们指导智能体的决策过程。

#### 策略（Policy）

策略定义了智能体在给定状态下应该采取什么动作。在语言模型的背景下，策略决定了模型在给定上下文下生成下一个词的概率分布。

策略可以分为两种主要类型：

1. **确定性策略（Deterministic Policy）**：
   - 对于每个状态，总是选择同一个动作
   - 形式上表示为：a = π(s)，其中a是动作，s是状态
   - 在语言生成中较少使用，因为它会导致缺乏多样性

2. **随机策略（Stochastic Policy）**：
   - 对于每个状态，定义动作的概率分布
   - 形式上表示为：π(a|s)，表示在状态s下选择动作a的概率
   - 语言模型通常使用随机策略，以保持生成的多样性

在强化学习微调中，我们通常从一个预训练的语言模型开始，该模型已经定义了一个初始策略（通常称为参考策略或行为策略）。然后，我们通过强化学习来优化这个策略，使其生成的内容能够获得更高的奖励。

#### 价值函数（Value Function）

价值函数估计在给定状态下，预期未来奖励的累积值。它帮助智能体评估不同状态的价值，从而做出更好的决策。

主要有两种类型的价值函数：

1. **状态价值函数（State Value Function）**：
   - 估计从状态s开始，遵循策略π行动所能获得的预期累积奖励
   - 表示为：V^π(s) = E_π[Σ γ^t * r_t | s_0 = s]
   - 其中γ是折扣因子，r_t是时间步t的奖励

2. **动作价值函数（Action Value Function）**：
   - 估计在状态s下选择动作a，然后遵循策略π行动所能获得的预期累积奖励
   - 表示为：Q^π(s,a) = E_π[Σ γ^t * r_t | s_0 = s, a_0 = a]
   - 也称为Q函数，是许多强化学习算法的基础

在语言生成的背景下，价值函数可以帮助模型评估不同生成选择的长期影响。例如，虽然某个词可能立即看起来很吸引人，但它可能导致故事后续发展受限，从而降低整体质量。价值函数可以帮助模型考虑这些长期影响。

#### 策略优化

强化学习的核心目标是找到一个最优策略，使预期累积奖励最大化。在语言模型的背景下，这意味着找到一个生成策略，使模型能够创作出获得最高评价的故事。

策略优化的主要方法包括：

1. **策略梯度（Policy Gradient）**：
   - 直接优化策略参数，使预期奖励最大化
   - 基于采样的轨迹计算梯度
   - 包括REINFORCE、A2C、PPO等算法
   - 适用于大型动作空间，因此常用于语言模型

2. **值迭代（Value Iteration）**：
   - 通过迭代更新价值函数来找到最优策略
   - 包括Q-learning、DQN等算法
   - 在大型离散动作空间中效率较低，较少用于语言模型

3. **演员-评论家（Actor-Critic）**：
   - 结合策略梯度和值函数学习
   - 演员（Actor）学习策略，评论家（Critic）学习值函数
   - 减少方差，提高样本效率
   - PPO是一种常用的演员-评论家算法，广泛应用于语言模型的强化学习

在实践中，策略优化通常需要解决几个关键挑战：

1. **探索与利用的平衡**：
   - 需要在探索新的生成可能性和利用已知的好策略之间取得平衡
   - 过度探索会导致不稳定性，过度利用会限制改进空间
   - 常用技术包括熵正则化、ε-贪心策略等

2. **样本效率**：
   - 生成和评估语言样本的成本很高
   - 需要高效利用每个样本
   - 离线强化学习和经验回放可以提高样本效率

3. **稳定性**：
   - 策略优化过程容易不稳定，特别是对于大型语言模型
   - 需要约束策略更新幅度，避免过度偏离初始策略
   - 这就是为什么PPO等约束优化算法在语言模型中很受欢迎

### 探索与利用的平衡

在强化学习中，探索（Exploration）与利用（Exploitation）的平衡是一个核心挑战。探索意味着尝试新的、未知的动作以发现潜在的更好策略；利用则是基于当前知识选择预期奖励最高的动作。

在故事生成的背景下，这个平衡尤为重要：

1. **过度探索的风险**：
   - 生成过于随机或实验性的内容
   - 可能导致不连贯或不相关的故事
   - 训练不稳定，难以收敛

2. **过度利用的风险**：
   - 生成过于保守或公式化的内容
   - 缺乏创意和多样性
   - 可能陷入局部最优，无法发现更好的叙事策略

为了在故事生成中平衡探索与利用，可以采用以下策略：

1. **熵正则化（Entropy Regularization）**：
   - 在优化目标中添加策略熵项
   - 鼓励策略保持一定的随机性
   - 防止过早收敛到确定性策略
   - 实现示例：

```python
def calculate_entropy_regularized_loss(logprobs, rewards, entropy, beta=0.01):
    """
    计算带熵正则化的策略梯度损失
    
    参数:
    - logprobs: 动作的对数概率
    - rewards: 相应的奖励
    - entropy: 策略熵
    - beta: 熵正则化系数
    
    返回:
    - loss: 正则化后的损失
    """
    policy_loss = -torch.mean(logprobs * rewards)
    entropy_bonus = -beta * entropy.mean()
    
    # 总损失 = 策略损失 - 熵奖励
    loss = policy_loss + entropy_bonus
    
    return loss
```

2. **KL散度约束（KL Divergence Constraint）**：
   - 限制新策略与参考策略之间的差异
   - 防止策略更新过大，导致不稳定
   - 保留预训练模型的语言能力
   - 这是PPO和DPO等算法的核心组成部分

3. **温度采样（Temperature Sampling）**：
   - 通过调整温度参数控制生成的随机性
   - 较高的温度增加探索，较低的温度增加利用
   - 可以在训练过程中动态调整温度
   - 实现示例：

```python
def temperature_sampling(logits, temperature=1.0):
    """
    使用温度参数进行采样
    
    参数:
    - logits: 模型输出的原始logits
    - temperature: 温度参数，控制随机性
    
    返回:
    - sampled_token_id: 采样的token ID
    """
    # 应用温度
    scaled_logits = logits / temperature
    
    # 计算概率分布
    probs = torch.softmax(scaled_logits, dim=-1)
    
    # 采样
    sampled_token_id = torch.multinomial(probs, 1).item()
    
    return sampled_token_id
```

4. **课程学习（Curriculum Learning）**：
   - 从简单任务开始，逐渐增加难度
   - 初期可以使用更多的探索，后期逐渐增加利用
   - 例如，先优化短故事，再逐渐过渡到长故事

5. **多目标优化（Multi-objective Optimization）**：
   - 同时优化多个目标，如奖励最大化和多样性
   - 使用帕累托优化或加权组合的方法
   - 可以更全面地评估生成策略的质量

在实践中，平衡探索与利用通常需要仔细调整超参数，并根据具体任务和模型特性进行适应。下一节中介绍的RLHF方法提供了一个更结构化的框架来处理这个平衡问题。

### RL在NLP中的应用挑战

虽然强化学习为优化语言模型提供了强大的框架，但在实际应用中仍面临一些特殊挑战：

1. **奖励黑客（Reward Hacking）**：
   - 模型可能学会欺骗奖励函数，而不是真正提高内容质量
   - 例如，如果奖励函数过度重视某些表面特征（如词汇多样性），模型可能会生成不自然但满足这些特征的文本
   - 解决方法：设计更全面的奖励函数，使用人类反馈，定期更新评估标准

2. **奖励稀疏性（Reward Sparsity）**：
   - 在故事生成中，有意义的奖励通常只在完整故事生成后才能获得
   - 这使得模型难以学习哪些中间决策是有价值的
   - 解决方法：奖励塑造（reward shaping）、中间奖励、价值函数学习

3. **分布偏移（Distribution Shift）**：
   - 强化学习可能导致模型生成分布偏离原始训练数据
   - 这可能导致语言质量下降或生成不自然的文本
   - 解决方法：KL散度约束、参考模型正则化、混合SFT和RL训练

4. **评估挑战（Evaluation Challenges）**：
   - 自动评估指标可能无法准确反映人类偏好
   - 人类评估成本高且可能不一致
   - 解决方法：多维度评估、结合自动和人工评估、持续改进评估方法

5. **计算效率（Computational Efficiency）**：
   - RL训练通常比SFT更计算密集
   - 需要多次前向和后向传播来评估和更新策略
   - 解决方法：批处理优化、分布式训练、参数高效微调技术

6. **样本效率（Sample Efficiency）**：
   - 传统RL方法需要大量样本才能收敛
   - 生成和评估语言样本的成本很高
   - 解决方法：离线RL、经验回放、模型辅助RL

为了应对这些挑战，研究人员开发了专门针对语言模型的RL方法，如RLHF和DPO，我们将在后续章节中详细讨论。这些方法通过结合人类偏好和约束优化，提供了更实用和稳定的训练框架。

### 实际应用示例

为了具体说明强化学习在故事生成中的应用，让我们考虑一个简化的实际案例：

#### 案例：优化童话故事的教育价值和吸引力

假设我们有一个通过SFT初步训练的故事讲述模型，现在我们希望通过RL进一步优化它，使其生成的童话故事既有教育价值又具吸引力。

**步骤1：定义奖励函数**

我们设计一个多维度奖励函数，包括：
- 教育价值（是否包含积极的道德教训）
- 吸引力（是否有趣、引人入胜）
- 适龄性（是否适合目标年龄段的儿童）
- 语言质量（是否流畅、连贯）

```python
def reward_function(story, target_age=8):
    # 教育价值评估（使用预训练的分类器）
    educational_value = educational_classifier.predict(story)
    
    # 吸引力评估（基于情感分析和叙事结构）
    engagement = engagement_analyzer.score(story)
    
    # 适龄性评估（基于词汇复杂度和主题适当性）
    age_appropriateness = age_classifier.predict(story, target_age)
    
    # 语言质量评估
    language_quality = language_evaluator.score(story)
    
    # 组合奖励（加权和）
    total_reward = (
        0.3 * educational_value +
        0.3 * engagement +
        0.2 * age_appropriateness +
        0.2 * language_quality
    )
    
    return total_reward
```

**步骤2：收集人类偏好数据**

为了训练更准确的奖励模型，我们收集人类对故事对的偏好：

1. 使用初始模型生成多个故事变体
2. 让人类评估者（如教师、家长）选择他们更喜欢的版本
3. 记录这些偏好对，用于训练奖励模型

```python
# 偏好数据示例
preference_data = [
    {
        "story_a": "从前，有一只勤劳的小蚂蚁...",
        "story_b": "很久以前，一只懒惰的小蚂蚁...",
        "chosen": "story_a",  # 人类评估者选择了故事A
        "prompt": "写一个关于勤劳的故事"
    },
    # 更多偏好对...
]
```

**步骤3：训练奖励模型**

使用收集的偏好数据训练一个奖励模型，该模型学习预测人类的偏好：

```python
def train_reward_model(preference_data, model, tokenizer, epochs=3):
    """训练奖励模型预测人类偏好"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # 获取故事对和偏好标签
            story_a = batch["story_a"]
            story_b = batch["story_b"]
            labels = batch["labels"]  # 1表示偏好A，0表示偏好B
            
            # 编码故事
            inputs_a = tokenizer(story_a, return_tensors="pt", padding=True, truncation=True)
            inputs_b = tokenizer(story_b, return_tensors="pt", padding=True, truncation=True)
            
            # 获取奖励分数
            reward_a = model(**inputs_a).logits
            reward_b = model(**inputs_b).logits
            
            # 计算损失（使用Bradley-Terry模型）
            probs = torch.sigmoid(reward_a - reward_b)
            loss = F.binary_cross_entropy(probs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}")
    
    return model
```

**步骤4：使用PPO进行强化学习训练**

使用训练好的奖励模型指导故事生成模型的优化：

```python
def train_with_ppo(model, reward_model, tokenizer, prompts, ppo_trainer, epochs=3):
    """使用PPO优化故事生成模型"""
    for epoch in range(epochs):
        for batch in dataloader:
            # 获取提示
            prompt_batch = batch["prompts"]
            
            # 生成故事
            generation_kwargs = {
                "max_length": 512,
                "min_length": 64,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9
            }
            
            # 使用PPO生成和优化
            response_tensors = []
            for prompt in prompt_batch:
                response = ppo_trainer.generate(prompt, **generation_kwargs)
                response_tensors.append(response)
            
            # 计算奖励
            rewards = []
            for response in response_tensors:
                story = tokenizer.decode(response)
                reward = reward_model(story)
                rewards.append(reward)
            
            # PPO更新步骤
            stats = ppo_trainer.step(prompt_batch, response_tensors, rewards)
            
            print(f"Epoch {epoch+1}, Mean Reward: {stats['ppo/mean_reward']}")
    
    return model
```

**步骤5：评估和迭代**

定期评估模型性能，并根据需要调整奖励函数或训练策略：

```python
def evaluate_model(model, test_prompts, human_evaluators=None):
    """评估模型生成的故事质量"""
    results = []
    
    for prompt in test_prompts:
        # 生成故事
        story = generate_story(model, prompt)
        
        # 自动评估
        auto_scores = {
            "educational_value": educational_classifier.predict(story),
            "engagement": engagement_analyzer.score(story),
            "language_quality": language_evaluator.score(story)
        }
        
        # 人工评估（如果可用）
        human_scores = {}
        if human_evaluators:
            fo<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>