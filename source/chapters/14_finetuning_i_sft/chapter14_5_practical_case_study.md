# 第14章：监督式微调 I: SFT-实践案例：故事讲述模型的SFT实现

## 14.5 实践案例：故事讲述模型的SFT实现

在前面的章节中，我们探讨了监督式微调的理论基础、参数高效微调技术、LoRA方法以及聊天模型的特殊考量。现在，让我们通过一个完整的实践案例，将这些知识整合起来，实现一个专门用于故事讲述的AI模型。本节将提供详细的实施步骤、代码示例和最佳实践，帮助读者构建自己的故事讲述AI。

### 数据集构建

构建高质量的故事数据集是成功微调的第一步。以下是构建故事讲述数据集的详细流程：

#### 数据来源选择

对于故事讲述任务，我们可以考虑以下数据来源：

1. **公开文学作品**：
   - 已进入公共领域的童话和短篇小说（如格林童话、安徒生童话）
   - Project Gutenberg等平台上的免费文学作品
   - 注意版权问题，确保使用合法

2. **专业创作内容**：
   - 委托专业作家创作的故事
   - 与出版商合作获取授权内容
   - 这类内容通常质量较高，但成本也更高

3. **合成数据**：
   - 使用现有大型语言模型生成初始故事
   - 人工编辑和审核，确保质量和适当性
   - 可以快速扩展数据集规模

4. **众包数据**：
   - 通过众包平台收集故事创作
   - 设计明确的指南和质量标准
   - 实施严格的质量控制流程

对于我们的案例，我们将使用混合方法：从公共领域收集经典童话，使用GPT-4生成现代故事变体，然后进行人工审核和编辑。

#### 数据处理流程

以下是处理原始故事文本的Python代码示例：

```python
import os
import re
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """清理文本，移除多余空白、特殊字符等"""
    # 替换多个空白为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊控制字符
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # 标准化引号
    text = text.replace('"', '"').replace('"', '"')
    # 修复常见的排版问题
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    return text.strip()

def split_into_sections(story: str, min_length: int = 200, max_length: int = 1000) -> List[str]:
    """将长故事分割成适当长度的段落"""
    # 按段落分割
    paragraphs = [p for p in story.split('\n\n') if p.strip()]
    
    sections = []
    current_section = ""
    
    for para in paragraphs:
        # 如果添加这个段落会超过最大长度，先保存当前部分
        if len(current_section) + len(para) > max_length and len(current_section) >= min_length:
            sections.append(current_section.strip())
            current_section = para
        else:
            if current_section:
                current_section += "\n\n" + para
            else:
                current_section = para
    
    # 添加最后一部分
    if current_section:
        sections.append(current_section.strip())
    
    return sections

def create_story_prompts(sections: List[str], title: str) -> List[Dict[str, str]]:
    """为故事段落创建提示-回应对"""
    prompt_response_pairs = []
    
    # 为第一部分创建开始提示
    first_prompt = f"请创作一个标题为"{title}"的故事。"
    prompt_response_pairs.append({
        "prompt": first_prompt,
        "response": sections[0]
    })
    
    # 为中间部分创建续写提示
    for i in range(1, len(sections)):
        continuation_prompt = f"请继续这个故事：\n\n{sections[i-1][-200:]}"
        prompt_response_pairs.append({
            "prompt": continuation_prompt,
            "response": sections[i]
        })
    
    return prompt_response_pairs

def process_story_file(file_path: str, output_dir: str) -> None:
    """处理单个故事文件，生成训练数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取标题（假设文件第一行是标题）
    lines = content.split('\n')
    title = lines[0].strip()
    story = '\n'.join(lines[1:])
    
    # 清理文本
    clean_story = clean_text(story)
    
    # 分割成段落
    sections = split_into_sections(clean_story)
    
    # 创建提示-回应对
    prompt_response_pairs = create_story_prompts(sections, title)
    
    # 保存为JSON文件
    output_file = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompt_response_pairs, f, ensure_ascii=False, indent=2)

def process_story_collection(input_dir: str, output_dir: str) -> None:
    """处理整个故事集合"""
    os.makedirs(output_dir, exist_ok=True)
    
    story_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    for file in tqdm(story_files, desc="处理故事文件"):
        process_story_file(os.path.join(input_dir, file), output_dir)
    
    print(f"已处理 {len(story_files)} 个故事文件，结果保存在 {output_dir}")

# 使用示例
if __name__ == "__main__":
    process_story_collection("raw_stories", "processed_stories")
```

#### 转换为对话格式

对于聊天模型，我们需要将故事数据转换为对话格式。以下是一个示例函数：

```python
def convert_to_chat_format(prompt_response_pairs: List[Dict[str, str]], 
                          system_prompt: str = "你是一个创意故事讲述者，擅长创作引人入胜的故事。") -> List[Dict[str, Any]]:
    """将提示-回应对转换为聊天格式"""
    chat_samples = []
    
    for pair in prompt_response_pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["response"]}
        ]
        
        chat_samples.append({"messages": messages})
    
    return chat_samples

def create_chat_dataset(input_dir: str, output_file: str) -> None:
    """创建聊天格式的数据集"""
    all_samples = []
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for file in tqdm(json_files, desc="转换为聊天格式"):
        with open(os.path.join(input_dir, file), 'r', encoding='utf-8') as f:
            prompt_response_pairs = json.load(f)
        
        chat_samples = convert_to_chat_format(prompt_response_pairs)
        all_samples.extend(chat_samples)
    
    # 保存为JSONL格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"已创建聊天数据集，包含 {len(all_samples)} 个样本，保存在 {output_file}")

# 使用示例
create_chat_dataset("processed_stories", "storyteller_chat_dataset.jsonl")
```

#### 数据集分割与验证

在训练前，我们需要将数据集分割为训练集和验证集，并进行最终验证：

```python
def split_and_validate_dataset(dataset_file: str, train_ratio: float = 0.9) -> None:
    """分割数据集并验证格式"""
    with open(dataset_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 打乱数据
    import random
    random.shuffle(lines)
    
    # 分割数据集
    split_idx = int(len(lines) * train_ratio)
    train_data = lines[:split_idx]
    val_data = lines[split_idx:]
    
    # 保存训练集
    train_file = dataset_file.replace('.jsonl', '_train.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    
    # 保存验证集
    val_file = dataset_file.replace('.jsonl', '_val.jsonl')
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_data)
    
    # 验证数据格式
    validation_errors = []
    for i, line in enumerate(train_data + val_data):
        try:
            sample = json.loads(line)
            # 检查必要字段
            if "messages" not in sample:
                validation_errors.append(f"样本 {i} 缺少 'messages' 字段")
                continue
            
            messages = sample["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                validation_errors.append(f"样本 {i} 的 'messages' 格式不正确")
                continue
            
            # 检查角色
            roles = [msg.get("role") for msg in messages]
            if "system" not in roles or "user" not in roles or "assistant" not in roles:
                validation_errors.append(f"样本 {i} 缺少必要的角色")
        except json.JSONDecodeError:
            validation_errors.append(f"样本 {i} JSON格式错误")
    
    if validation_errors:
        print(f"发现 {len(validation_errors)} 个验证错误:")
        for err in validation_errors[:10]:  # 只显示前10个错误
            print(f"  - {err}")
        if len(validation_errors) > 10:
            print(f"  ... 以及 {len(validation_errors) - 10} 个其他错误")
    else:
        print("数据集验证通过，未发现错误")
    
    print(f"训练集: {len(train_data)} 样本，保存在 {train_file}")
    print(f"验证集: {len(val_data)} 样本，保存在 {val_file}")

# 使用示例
split_and_validate_dataset("storyteller_chat_dataset.jsonl")
```

### 模型选择与配置

选择合适的基础模型对于故事讲述AI至关重要。以下是一些考虑因素：

1. **模型规模**：
   - 小型模型（1B-7B参数）：训练资源需求低，但生成能力有限
   - 中型模型（7B-13B参数）：平衡资源需求和生成质量，适合大多数应用
   - 大型模型（13B+参数）：生成质量高，但资源需求大

2. **开源许可**：
   - 确保模型许可允许商业使用（如适用）
   - 常见选择包括Llama 2、Mistral、Falcon等

3. **预训练数据**：
   - 选择在文学作品和创意写作上有良好表现的模型
   - 考虑模型的语言覆盖范围（如果需要多语言支持）

4. **社区支持**：
   - 活跃的社区意味着更多的资源和改进
   - 检查模型的更新频率和维护状态

对于我们的故事讲述AI，我们选择Llama 2 7B作为基础模型，因为它在创意写作方面表现良好，资源需求适中，且许可允许商业使用。

### 训练过程与参数设置

下面是使用LoRA进行微调的完整训练脚本：

```python
import os
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset

# 配置参数
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # 基础模型
OUTPUT_DIR = "./storyteller_model"       # 输出目录
TRAIN_FILE = "storyteller_chat_dataset_train.jsonl"  # 训练数据
VAL_FILE = "storyteller_chat_dataset_val.jsonl"      # 验证数据

# LoRA配置
LORA_R = 16                # LoRA秩
LORA_ALPHA = 32            # LoRA alpha参数
LORA_DROPOUT = 0.05        # LoRA dropout
TARGET_MODULES = [         # 目标模块
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]

# 训练参数
LEARNING_RATE = 2e-4       # 学习率
BATCH_SIZE = 4             # 批次大小
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
MAX_STEPS = 1000           # 最大训练步数
WARMUP_STEPS = 100         # 预热步数
LOGGING_STEPS = 10         # 日志记录间隔
SAVE_STEPS = 200           # 保存间隔
MAX_SEQ_LENGTH = 2048      # 最大序列长度

# 设置设备
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 配置量化参数（使用4位量化减少内存需求）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map=device_map,
)

# 准备模型进行训练
model = prepare_model_for_kbit_training(model)

# 配置LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 打印可训练参数信息
model.print_trainable_parameters()

# 数据处理函数
def preprocess_function(examples):
    """处理数据集样本为模型输入格式"""
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for sample in examples["messages"]:
        # 构建对话格式
        conversation = []
        for message in sample:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                conversation.append(f"<|system|>\n{content}")
            elif role == "user":
                conversation.append(f"<|user|>\n{content}")
            elif role == "assistant":
                conversation.append(f"<|assistant|>\n{content}")
        
        # 添加结束标记
        conversation.append("<|endoftext|>")
        
        # 连接所有消息
        text = "\n".join(conversation)
        
        # 分词
        tokenized = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)
        
        # 添加到结果
        result["input_ids"].append(tokenized["input_ids"])
        result["attention_mask"].append(tokenized["attention_mask"])
        result["labels"].append(tokenized["input_ids"].copy())
    
    return result

# 加载数据集
train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
val_dataset = load_dataset("json", data_files=VAL_FILE, split="train")

# 处理数据集
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
)

# 数据收集器
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True,
)

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_steps=MAX_STEPS,
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    evaluation_strategy="steps",
    eval_steps=SAVE_STEPS,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    weight_decay=0.05,
    save_total_limit=3,
    load_best_model_at_end=True,
    ddp_find_unused_parameters=False if ddp else None,
    report_to="tensorboard",
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model(OUTPUT_DIR)
```

### 评估与优化

训练完成后，我们需要评估模型性能并进行优化。以下是一些评估方法：

#### 自动评估指标

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from tqdm import tqdm
import evaluate

# 加载模型
base_model_name = "meta-llama/Llama-2-7b-hf"
adapter_path = "./storyteller_model"

# 加载配置
config = PeftConfig.from_pretrained(adapter_path)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(model, adapter_path)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 加载评估数据集
eval_dataset = load_dataset("json", data_files="storyteller_chat_dataset_val.jsonl", split="train")

# 加载评估指标
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

def generate_story(prompt, max_length=1024):
    """生成故事"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 评估函数
def evaluate_model(dataset, num_samples=100):
    """评估模型性能"""
    # 限制样本数量
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    rouge_scores = []
    meteor_scores = []
    
    for sample in tqdm(dataset, desc="Evaluating"):
        messages = sample["messages"]
        
        # 提取系统提示和用户输入
        system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_prompt = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        
        # 提取参考回答
        reference = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
        
        # 构建完整提示
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # 生成故事
        generated = generate_story(full_prompt)
        
        # 计算ROUGE分数
        rouge_result = rouge.compute(predictions=[generated], references=[reference])
        rouge_scores.append(rouge_result)
        
        # 计算METEOR分数
        meteor_result = meteor.compute(predictions=[generated], references=[reference])
        meteor_scores.append(meteor_result["meteor"])
    
    # 计算平均分数
    avg_rouge = {k: np.mean([score[k] for score in rouge_scores]) for k in rouge_scores[0].keys()}
    avg_meteor = np.mean(meteor_scores)
    
    return {
        "rouge": avg_rouge,
        "meteor": avg_meteor
    }

# 运行评估
results = evaluate_model(eval_dataset)
print("评估结果:")
print(f"ROUGE-1: {results['rouge']['rouge1']:.4f}")
print(f"ROUGE-2: {results['rouge']['rouge2']:.4f}")
print(f"ROUGE-L: {results['rouge']['rougeL']:.4f}")
print(f"METEOR: {results['meteor']:.4f}")
```

#### 人工评估

自动指标无法完全捕捉故事的创造性和吸引力，因此人工评估至关重要。以下是一个人工评估表格示例：

```python
import pandas as pd
import random
from IPython.display import display, HTML

def create_human_evaluation_form(dataset, model, num_samples=10):
    """创建人工评估表格"""
    # 随机选择样本
    sample_indices = random.sample(range(len(dataset)), num_samples)
    selected_samples = [dataset[i] for i in sample_indices]
    
    evaluation_data = []
    
    for i, sample in enumerate(selected_samples):
        messages = sample["messages"]
        
        # 提取系统提示和用户输入
        system_prompt = next((msg["conte<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>