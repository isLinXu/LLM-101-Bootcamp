# 第6章：分词技术(Tokenization)

## 6.1 分词的基本概念与重要性

在构建故事讲述AI大语言模型的过程中，分词（Tokenization）是我们需要面对的第一个关键技术挑战。分词是将原始文本转换为模型可以处理的数字序列的过程，它是连接人类语言和机器语言的桥梁。无论多么复杂的语言模型，其本质上都是对数字序列而非直接对文本进行操作的。

分词的重要性不言而喻。首先，它直接影响模型的词汇量和表达能力。一个设计良好的分词器能够用有限的词元（token）表示丰富的语言现象，使模型能够理解和生成多样化的文本。其次，分词策略会影响模型的训练效率和推理速度。合理的分词可以减少序列长度，降低计算复杂度。最后，分词还会影响模型对不同语言、特殊符号和罕见词的处理能力，这对于构建一个通用的故事生成模型尤为重要。

在早期的自然语言处理中，分词通常基于简单的规则，如按空格分割英文单词或使用字典匹配中文词语。然而，这些方法难以处理未登录词（out-of-vocabulary words）和跨语言场景。现代大语言模型普遍采用的是基于统计的子词（subword）分词方法，其中最具代表性的就是字节对编码（Byte Pair Encoding, BPE）及其变种。

## 6.2 字节对编码(Byte Pair Encoding, BPE)原理

字节对编码最初是一种数据压缩算法，由Gage于1994年提出。在自然语言处理领域，BPE被Sennrich等人于2016年引入用于神经机器翻译的分词任务，随后在GPT、BERT等大语言模型中得到广泛应用。

BPE的核心思想非常直观：频繁一起出现的字符序列应该被视为一个整体。具体来说，BPE算法首先将文本分解为最小单位（通常是单个字符或字节），然后迭代地合并最频繁出现的相邻符号对，直到达到预设的词汇量或无法找到频率超过阈值的符号对为止。

让我们通过一个简单的例子来理解BPE的工作原理：

假设我们有以下训练语料：
```
低碳生活 低碳出行 低碳饮食
```

1. 初始化词汇表为单个字符：{'低', '碳', '生', '活', '出', '行', '饮', '食', ' '}
2. 统计所有相邻字符对的频率：
   - ('低', '碳'): 3次
   - ('碳', ' '): 1次
   - (' ', '生'): 1次
   - ('生', '活'): 1次
   - (' ', '出'): 1次
   - ('出', '行'): 1次
   - (' ', '饮'): 1次
   - ('饮', '食'): 1次
   - ('碳', '生'): 1次
   - ('碳', '出'): 1次
   - ('碳', '饮'): 1次
3. 合并最频繁的字符对('低', '碳')为新符号'低碳'
4. 更新词汇表：{'低', '碳', '生', '活', '出', '行', '饮', '食', ' ', '低碳'}
5. 更新语料：
   ```
   低碳生活 低碳出行 低碳饮食
   ```
6. 继续迭代，直到达到预设的词汇量或合并条件不再满足

通过这个过程，BPE算法能够自动发现语料中的常见模式，并将其作为独立的词元。这使得模型可以高效地表示常见词，同时保留处理罕见词和未见词的能力（通过将其分解为更小的子词单元）。

## 6.3 minBPE算法详解

minBPE是BPE算法的一个简化和优化版本，由Jurafsky和Martin在他们的自然语言处理教材中提出。它保留了BPE的核心思想，但在实现上更加高效和易于理解。

minBPE算法的主要步骤如下：

1. **初始化**：将训练语料分解为最小单位（通常是Unicode字符或字节），并统计每个单位的频率。
2. **构建初始词汇表**：初始词汇表包含所有出现在语料中的基本单位。
3. **迭代合并**：
   a. 统计所有相邻词元对的频率
   b. 选择频率最高的词元对进行合并
   c. 将新合并的词元添加到词汇表中
   d. 更新语料中的词元表示
   e. 重复上述步骤，直到达到预设的词汇量或满足停止条件
4. **构建编码器和解码器**：基于最终的词汇表，构建从文本到词元ID的映射（编码器）和从词元ID到文本的映射（解码器）。

minBPE相比原始BPE的主要优化在于：
- 使用更高效的数据结构来跟踪词元对的频率
- 简化了合并过程中的语料更新操作
- 提供了更清晰的编码和解码接口

下面是一个简化的minBPE实现示例：

```python
def train_minbpe(corpus, vocab_size):
    # 初始化词汇表为单个字符
    vocab = list(set(''.join(corpus)))
    
    # 将语料转换为字符列表的列表
    corpus_tokens = [[c for c in text] for text in corpus]
    
    # 迭代合并直到达到目标词汇量
    while len(vocab) < vocab_size:
        # 统计所有相邻词元对的频率
        pairs = {}
        for text in corpus_tokens:
            for i in range(len(text) - 1):
                pair = (text[i], text[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        
        # 如果没有可合并的词元对，提前结束
        if not pairs:
            break
        
        # 选择频率最高的词元对
        best_pair = max(pairs, key=pairs.get)
        new_token = ''.join(best_pair)
        
        # 将新词元添加到词汇表
        vocab.append(new_token)
        
        # 更新语料中的词元表示
        for i, text in enumerate(corpus_tokens):
            j = 0
            while j < len(text) - 1:
                if text[j] == best_pair[0] and text[j + 1] == best_pair[1]:
                    text[j] = new_token
                    text.pop(j + 1)
                else:
                    j += 1
    
    return vocab
```

这个简化实现展示了minBPE的核心逻辑，但在实际应用中，我们通常需要更多的优化和功能，如处理未知字符、支持特殊标记（如开始和结束标记）、处理空格和标点符号等。

## 6.4 实现一个简单的分词器

现在，让我们实现一个完整的minBPE分词器，包括训练、编码和解码功能。这个分词器将能够处理中英文混合文本，并支持基本的特殊标记。

```python
import re
from collections import defaultdict

class MinBPETokenizer:
    def __init__(self):
        self.encoder = {}  # 从词元到ID的映射
        self.decoder = {}  # 从ID到词元的映射
        self.vocab = []    # 词汇表
        self.special_tokens = {
            "<PAD>": 0,    # 填充标记
            "<BOS>": 1,    # 序列开始标记
            "<EOS>": 2,    # 序列结束标记
            "<UNK>": 3     # 未知词标记
        }
    
    def train(self, corpus, vocab_size=1000, min_frequency=2):
        """训练分词器"""
        # 预处理语料
        processed_corpus = [self._preprocess_text(text) for text in corpus]
        
        # 初始化词汇表为单个字符
        chars = set()
        for text in processed_corpus:
            chars.update(text)
        self.vocab = list(self.special_tokens.keys()) + list(chars)
        
        # 将语料转换为字符列表的列表
        corpus_tokens = [[c for c in text] for text in processed_corpus]
        
        # 迭代合并直到达到目标词汇量
        while len(self.vocab) < vocab_size:
            # 统计所有相邻词元对的频率
            pairs = defaultdict(int)
            for text in corpus_tokens:
                for i in range(len(text) - 1):
                    pair = (text[i], text[i + 1])
                    pairs[pair] += 1
            
            # 过滤低频词元对
            pairs = {pair: freq for pair, freq in pairs.items() if freq >= min_frequency}
            
            # 如果没有可合并的词元对，提前结束
            if not pairs:
                break
            
            # 选择频率最高的词元对
            best_pair = max(pairs, key=pairs.get)
            new_token = ''.join(best_pair)
            
            # 将新词元添加到词汇表
            self.vocab.append(new_token)
            
            # 更新语料中的词元表示
            for i, text in enumerate(corpus_tokens):
                j = 0
                while j < len(text) - 1:
                    if text[j] == best_pair[0] and text[j + 1] == best_pair[1]:
                        text[j] = new_token
                        text.pop(j + 1)
                    else:
                        j += 1
        
        # 构建编码器和解码器
        self.encoder = {token: i for i, token in enumerate(self.vocab)}
        self.decoder = {i: token for i, token in enumerate(self.vocab)}
        
        return self
    
    def encode(self, text):
        """将文本编码为词元ID序列"""
        text = self._preprocess_text(text)
        
        # 贪婪分词
        tokens = []
        i = 0
        while i < len(text):
            # 尝试匹配最长的词元
            matched = False
            for j in range(min(100, len(text) - i), 0, -1):  # 限制最大词元长度为100
                if text[i:i+j] in self.encoder:
                    tokens.append(self.encoder[text[i:i+j]])
                    i += j
                    matched = True
                    break
            
            # 如果没有匹配到任何词元，使用未知词标记
            if not matched:
                tokens.append(self.special_tokens["<UNK>"])
                i += 1
        
        return tokens
    
    def decode(self, ids):
        """将词元ID序列解码为文本"""
        tokens = [self.decoder.get(id, "<UNK>") for id in ids]
        text = ''.join(tokens)
        return text
    
    def _preprocess_text(self, text):
        """预处理文本"""
        # 可以添加各种预处理步骤，如小写化、规范化等
        return text
```

这个实现包含了一个完整的minBPE分词器，具有以下功能：
- 支持从语料库训练词汇表
- 支持特殊标记（填充、序列开始/结束、未知词）
- 提供编码（文本到ID）和解码（ID到文本）功能
- 使用贪婪算法进行分词，优先匹配最长的词元

在实际应用中，我们可能还需要添加更多功能，如保存和加载模型、批处理、多线程训练等。但这个基本实现已经足够用于理解分词的核心原理和实现方法。

## 6.5 分词器训练与优化

在实际应用中，分词器的训练和优化是一个复杂的过程，需要考虑多种因素。以下是一些关键的优化策略和最佳实践：

### 6.5.1 词汇量选择

词汇量（vocabulary size）是分词器最重要的超参数之一。词汇量过小会导致分词过于细碎，增加序列长度并降低模型效率；词汇量过大则会增加模型参数量，并可能导致数据稀疏问题。

对于英文为主的语料，通常的词汇量在30,000到50,000之间；对于多语言模型，词汇量可能需要更大，如100,000或更多。在我们的故事生成模型中，由于需要处理多种语言和创意文本，建议使用50,000左右的词汇量。

### 6.5.2 训练语料选择

分词器的性能很大程度上取决于训练语料的质量和多样性。理想的训练语料应该：
- 覆盖模型将要处理的所有语言和领域
- 包含足够的罕见词和特殊表达
- 在不同类型的文本（如对话、叙述、描述）之间保持平衡
- 具有足够的规模（通常需要几GB的文本）

对于故事生成模型，我们应该收集各种类型的故事、小说、对话和描述性文本，确保语料的多样性和代表性。

### 6.5.3 预处理策略

在训练分词器之前，对语料进行适当的预处理可以显著提高分词质量：
- 规范化：统一空格、标点符号和特殊字符的表示
- 大小写处理：根据任务需求决定是否保留大小写信息
- 去除噪声：过滤掉HTML标签、重复文本等噪声
- 分段：将长文本分割为适当长度的段落

对于故事生成，保留原始的大小写、标点和段落结构通常很重要，因为这些元素对故事的风格和可读性有重要影响。

### 6.5.4 高级BPE变种

除了基本的minBPE算法，还有几种高级变种值得考虑：

**1. WordPiece**

WordPiece是Google在BERT中使用的分词算法，它与BPE类似，但使用不同的合并标准。WordPiece基于语言模型似然而非简单频率来选择合并的词元对，这有助于生成更有语言学意义的子词。

**2. Unigram Language Model**

Unigram是另一种流行的子词分词算法，它基于概率模型而非确定性规则。Unigram首先初始化一个大词汇表，然后迭代地移除对语料编码贡献最小的词元，直到达到目标词汇量。这种方法通常能产生更平衡的分词结果。

**3. SentencePiece**

SentencePiece是一个综合性的分词工具，支持BPE和Unigram算法，并具有以下特点：
- 将空格视为普通字符，支持无空格语言（如中文、日文）
- 直接从原始文本学习，无需语言特定的预处理
- 支持字节级别的回退，确保任何文本都可以编码

对于多语言故事生成模型，SentencePiece是一个很好的选择，因为它能够一致地处理不同语言的文本。

### 6.5.5 性能优化

在实际应用中，分词器的性能（速度和内存使用）也是一个重要考虑因素：

**1. 并行处理**

使用多线程或多进程并行处理大规模语料，加速训练过程。

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_train(corpus_chunks, vocab_size_per_chunk):
    with ProcessPoolExecutor() as executor:
        tokenizers = list(executor.map(
            lambda chunk: MinBPETokenizer().train(chunk, vocab_size_per_chunk),
            corpus_chunks
        ))
    
    # 合并多个分词器的词汇表
    merged_vocab = []
    for tokenizer in tokenizers:
        merged_vocab.extend(tokenizer.vocab)
    
    # 去重并选择最终词汇表
    final_vocab = list(dict.fromkeys(merged_vocab))[:vocab_size]
    
    # 创建最终分词器
    final_tokenizer = MinBPETokenizer()
    final_tokenizer.vocab = final_vocab
    final_tokenizer.encoder = {token: i for i, token in enumerate(final_vocab)}
    final_tokenizer.decoder = {i: token for i, token in enumerate(final_vocab)}
    
    return final_tokenizer
```

**2. 高效数据结构**

使用高效的数据结构来存储和查询词汇表，如前缀树（Trie）或哈希表。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.token_id = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, token, token_id):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.token_id = token_id
    
    def find_longest_prefix(self, text, start_pos):
        node = self.root
        longest_match = None
        longest_id = None
        match_length = 0
        
        for i in range(start_pos, len(text)):
            char = text[i]
            if char not in node.children:
                break
            
            node = node.children[char]
            match_length += 1
            
            if node.is_end:
                longest_match = text[start_pos:start_pos + match_length]
                longest_id = node.token_id
        
        return longest_match, longest_id, match_length
```

**3. 缓存机制**

对于常见的文本片段，使用缓存来避免重复计算。

```python
class CachedTokenizer:
    def __init__(self, base_tokenizer, cache_size=10000):
        self.base_tokenizer = base_tokenizer
        self.cache = {}
        self.cache_size = cache_size
    
    def encode(self, text):
        if text in self.cache:
            return self.cache[text]
        
        tokens = self.base_tokenizer.encode(text)
        
        # 简单的LRU缓存管理
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[text] = tokens
        return tokens
    
    def decode(self, ids):
        # 解码通常不需要缓存，因为解码操作相对简单
        return self.base_tokenizer.decode(ids)
```

## 6.6 在故事生成中的应用

分词技术在故事生成中有着广泛的应用，它不仅影响模型的基本功能，还可以用来增强故事的质量和多样性。

### 6.6.1 多语言故事生成

一个好的分词器能够处理多种语言的文本，使模型能够生成多语言故事或在故事中自然地混合不同语言。例如，一个角色可能说一句外语，或者故事可能包含不同文化背景的名称和术语。

对于多语言支持，我们可以扩展之前的分词器实现：

```python
class MultilingualTokenizer(MinBPETokenizer):
    def __init__(self):
        super().__init__()
        self.language_specific_preprocessing = {
            'en': self._preprocess_english,
            'zh': self._preprocess_chinese,
            # 可以添加更多语言的预处理函数
        }
    
    def _preprocess_english(self, text):
        # 英文特定的预处理，如处理缩写、标点等
        text = re.sub(r"([.!?])", r" \1", text)  # 在标点符号前添加空格
        text = re.sub(r"[^a-zA-Z0-9.!?]+", " ", text)  # 规范化空格
        return text.lower()  # 小写化
    
    def _preprocess_chinese(self, text):
        # 中文特定的预处理，如处理全角符号等
        text = re.sub(r"[【】「」『』]", "", text)  # 移除特定括号
        return text  # 中文通常不需要小写化
    
    def _detect_language(self, text):
        # 简单的语言检测，实际应用中可能需要更复杂的算法
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'  # 包含汉字，判断为中文
        else:
            return 'en'  # 默认为英文
    
    def _preprocess_text(self, text):
        # 根据检测到的语言选择相应的预处理函数
        lang = self._detect_language(text)
        if lang in self.language_specific_preprocessing:
            return self.language_specific_preprocessing[lang](text)
        else:
            return super()._preprocess_text(text)
```

### 6.6.2 风格和语调控制

分词器可以帮助模型识别和生成特定风格的文本。例如，通过在训练数据中标记不同风格的文本，模型可以学会根据提示生成不同风格的故事。

```python
def add_style_tokens(tokenizer, styles=['formal', 'casual', 'poetic', 'humorous']):
    """向分词器添加风格标记"""
    for style in styles:
        style_token = f"<{style}>"
        if style_token not in tokenizer.encoder:
            tokenizer.vocab.append(style_token)
            token_id = len(tokenizer.encoder)
            tokenizer.encoder[style_token] = token_id
            tokenizer.decoder[token_id] = style_token
    
    return tokenizer

# 使用风格标记
def encode_with_style(tokenizer, text, style='formal'):
    """使用指定风格标记编码文本"""
    style_token = f"<{style}>"
    style_id = tokenizer.encoder.get(style_token)
    if style_id is None:
        raise ValueError(f"未知的风格: {style}")
    
    # 在文本开头添加风格标记
    ids = [style_id] + tokenizer.encode(text)
    return ids
```

### 6.6.3 角色语音和对话

在故事中，不同角色可能有不同的说话方式。通过特殊的角色标记，模型可以学会模仿特定角色的语音和对话风格。

```python
def add_character_tokens(tokenizer, characters=['narrator', 'protagonist', 'antagonist']):
    """向分词器添加角色标记"""
    for character in characters:
        char_token = f"<{character}>"
        if char_token not in tokenizer.encoder:
            tokenizer.vocab.append(char_token)
            token_id = len(tokenizer.encoder)
            tokenizer.encoder[char_token] = token_id
            tokenizer.decoder[token_id] = char_token
    
    return tokenizer

# 使用角色标记处理对话
def process_dialogue(tokenizer, dialogue, character='narrator'):
    """处理特定角色的对话"""
    char_token = f"<{character}>"
    char_id = tokenizer.encoder.get(char_token)
    if char_id is None:
        raise ValueError(f"未知的角色: {character}")
    
    # 在对话开头添加角色标记
    ids = [char_id] + tokenizer.encode(dialogue)
    return ids
```

### 6.6.4 情感和氛围控制

类似地，我们可以使用特殊标记来控制故事的情感和氛围，使模型能够生成符合特定情感基调的故事。

```python
def add_emotion_tokens(tokenizer, emotions=['happy', 'sad', 'tense', 'mysterious']):
    """向分词器添加情感标记"""
    for emotion in emotions:
        emotion_token = f"<{emotion}>"
        if emotion_token not in tokenizer.encoder:
            tokenizer.vocab.append(emotion_token)
            token_id = len(tokenizer.encoder)
            tokenizer.encoder[emotion_token] = token_id
            tokenizer.decoder[token_id] = emotion_token
    
    return tokenizer

# 使用情感标记
def encode_with_emotion(tokenizer, text, emotion='happy'):
    """使用指定情感标记编码文本"""
    emotion_token = f"<{emotion}>"
    emotion_id = tokenizer.encoder.get(emotion_token)
    if emotion_id is None:
        raise ValueError(f"未知的情感: {emotion}")
    
    # 在文本开头添加情感标记
    ids = [emotion_id] + tokenizer.encode(text)
    return ids
```

### 6.6.5 故事结构和节奏

分词器还可以帮助模型理解和生成具有特定结构和节奏的故事。通过添加表示故事结构元素（如开场、高潮、结局）的特殊标记，模型可以学会构建结构完整的故事。

```python
def add_structure_tokens(tokenizer, elements=['intro', 'rising_action', 'climax', 'falling_action', 'resolution']):
    """向分词器添加故事结构标记"""
    for element in elements:
        structure_token = f"<{element}>"
        if structure_token not in tokenizer.encoder:
            tokenizer.vocab.append(structure_token)
            token_id = len(tokenizer.encoder)
            tokenizer.encoder[structure_token] = token_id
            tokenizer.decoder[token_id] = structure_token
    
    return tokenizer

# 使用故事结构标记
def encode_story_section(tokenizer, text, section='intro'):
    """使用指定故事结构标记编码文本段落"""
    section_token = f"<{section}>"
    section_id = tokenizer.encoder.get(section_token)
    if section_id is None:
        raise ValueError(f"未知的故事结构元素: {section}")
    
    # 在文本开头添加结构标记
    ids = [section_id] + tokenizer.encode(text)
    return ids
```

## 6.7 总结与展望

在本章中，我们深入探讨了分词技术的原理和实现，特别是字节对编码（BPE）及其变种minBPE算法。我们实现了一个基本的分词器，并讨论了如何优化它以适应故事生成的需求。我们还探索了分词技术在多语言支持、风格控制、角色对话、情感氛围和故事结构方面的应用。

分词是构建大语言模型的基础步骤，它直接影响模型的表达能力和效率。一个设计良好的分词器可以帮助模型更好地理解和生成自然、流畅、多样化的故事。

在接下来的章节中，我们将基于这个基础，探索如何优化模型训练过程，提高训练速度和效率，并最终构建一个完整的故事讲述AI系统。

**练习与思考**

1. 尝试使用本章实现的minBPE分词器处理一段中英文混合的故事文本，观察分词结果，并思考如何改进。
2. 比较不同词汇量（如1000、5000、10000）对分词结果的影响，讨论在故事生成场景中的最佳词汇量选择。
3. 设计一个实验，比较基本BPE、WordPiece和Unigram算法在故事文本上的性能差异。
4. 思考如何扩展分词器以支持更多语言，特别是结构差异较大的语言（如阿拉伯语、日语等）。
5. 探索如何使用分词技术来增强故事的创意性和多样性，例如通过特殊标记控制故事的风格、情感和结构。

**参考资料**

1. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics.
2. Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations.
3. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Blog.
4. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems.
5. Jurafsky, D., & Martin, J. H. (2021). Speech and Language Processing (3rd ed. draft). Chapter on Subword Models.
