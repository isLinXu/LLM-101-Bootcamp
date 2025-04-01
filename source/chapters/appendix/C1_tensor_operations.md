---
file_format: mystnb
kernelspec:
  name: python3
---
# 附录C：张量操作基础

## C.1 张量：形状、视图、步长与连续性

在深度学习和神经网络编程中，张量是最基本也是最重要的数据结构。张量可以看作是多维数组，是标量（0维张量）、向量（1维张量）和矩阵（2维张量）的泛化形式。本节将深入探讨张量的核心概念，包括形状、视图、步长和连续性，以及这些概念在构建高效神经网络中的应用。

### 张量的基本概念

张量是具有统一类型的多维数组，可以在不同维度上进行索引。在深度学习中，张量通常用于表示数据批次、特征映射、模型权重等。

#### 张量的维度和形状

张量的维度（也称为秩或阶）是指张量的轴数。张量的形状是一个元组，指定了张量在每个维度上的大小。

1. **0维张量（标量）**：
   - 单个数值
   - 形状为空元组 `()`
   - 例如：温度值 `42.0`

2. **1维张量（向量）**：
   - 数值序列
   - 形状为 `(n,)`，其中 `n` 是元素数量
   - 例如：一维数组 `[1, 2, 3, 4, 5]`，形状为 `(5,)`

3. **2维张量（矩阵）**：
   - 数值表格
   - 形状为 `(m, n)`，其中 `m` 是行数，`n` 是列数
   - 例如：2x3矩阵 `[[1, 2, 3], [4, 5, 6]]`，形状为 `(2, 3)`

4. **3维张量**：
   - 数值立方体
   - 形状为 `(l, m, n)`
   - 例如：RGB图像可以表示为形状为 `(height, width, 3)` 的3维张量

5. **4维张量**：
   - 形状为 `(k, l, m, n)`
   - 例如：批次的RGB图像可以表示为形状为 `(batch_size, height, width, 3)` 的4维张量

6. **更高维张量**：
   - 在复杂的深度学习模型中常见
   - 例如：视频数据可以表示为5维张量 `(batch_size, frames, height, width, channels)`

以下是使用PyTorch创建不同维度张量的示例：

```python
import torch

# 0维张量（标量）
scalar = torch.tensor(42.0)
print(f"Scalar: {scalar}, Shape: {scalar.shape}")
# 输出: Scalar: tensor(42.), Shape: torch.Size([])

# 1维张量（向量）
vector = torch.tensor([1, 2, 3, 4, 5])
print(f"Vector: {vector}, Shape: {vector.shape}")
# 输出: Vector: tensor([1, 2, 3, 4, 5]), Shape: torch.Size([5])

# 2维张量（矩阵）
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Matrix: {matrix}, Shape: {matrix.shape}")
# 输出: Matrix: tensor([[1, 2, 3], [4, 5, 6]]), Shape: torch.Size([2, 3])

# 3维张量
tensor_3d = torch.zeros((2, 3, 4))
print(f"3D Tensor Shape: {tensor_3d.shape}")
# 输出: 3D Tensor Shape: torch.Size([2, 3, 4])

# 4维张量
tensor_4d = torch.zeros((2, 3, 4, 5))
print(f"4D Tensor Shape: {tensor_4d.shape}")
# 输出: 4D Tensor Shape: torch.Size([2, 3, 4, 5])
```

#### 张量的数据类型

张量的数据类型定义了其元素的类型。常见的张量数据类型包括：

1. **浮点类型**：
   - `torch.float32` 或 `torch.float`：32位浮点数（默认）
   - `torch.float64` 或 `torch.double`：64位浮点数
   - `torch.float16` 或 `torch.half`：16位浮点数

2. **整数类型**：
   - `torch.int32` 或 `torch.int`：32位整数
   - `torch.int64` 或 `torch.long`：64位整数（默认整数类型）
   - `torch.int16` 或 `torch.short`：16位整数
   - `torch.int8`：8位整数

3. **布尔类型**：
   - `torch.bool`：布尔值（True或False）

4. **复数类型**：
   - `torch.complex64`：32位实部和虚部
   - `torch.complex128`：64位实部和虚部

以下是使用不同数据类型创建张量的示例：

```python
import torch

# 浮点类型
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
double_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
half_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)

# 整数类型
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
long_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
short_tensor = torch.tensor([1, 2, 3], dtype=torch.int16)

# 布尔类型
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)

# 复数类型
complex_tensor = torch.tensor([1+2j, 3+4j, 5+6j], dtype=torch.complex64)

# 打印数据类型
print(f"Float Tensor dtype: {float_tensor.dtype}")
print(f"Bool Tensor dtype: {bool_tensor.dtype}")
print(f"Complex Tensor dtype: {complex_tensor.dtype}")
```

### 张量的存储和布局

理解张量的内存布局对于高效的张量操作至关重要。张量的内存布局由其存储、步长和连续性决定。

#### 张量的存储

张量的存储是一个一维数组，包含张量的所有元素。张量的形状和步长定义了如何将这个一维数组解释为多维张量。

```python
import torch

# 创建一个2x3矩阵
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 查看存储
print(f"Matrix: {matrix}")
print(f"Storage: {matrix.storage()}")
# 在PyTorch中，可以通过.storage()方法查看张量的底层存储
# 输出类似于: Storage: 1 2 3 4 5 6
```

#### 张量的步长

张量的步长（stride）是一个元组，指定了在每个维度上前进一步需要跳过的元素数量。步长决定了张量元素在内存中的布局方式。

```python
import torch

# 创建一个2x3矩阵
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 查看形状和步长
print(f"Matrix: {matrix}")
print(f"Shape: {matrix.shape}")
print(f"Stride: {matrix.stride()}")
# 输出: Stride: (3, 1)
# 这意味着在第0维（行）上前进一步需要跳过3个元素，
# 在第1维（列）上前进一步需要跳过1个元素
```

步长的概念可以通过以下方式理解：

1. 对于形状为 `(m, n)` 的行优先（row-major）矩阵，步长通常为 `(n, 1)`
2. 对于形状为 `(m, n)` 的列优先（column-major）矩阵，步长通常为 `(1, m)`

步长的值反映了在特定维度上移动一个单位时，在线性内存中需要跳过的元素数量。

#### 张量的连续性

张量的连续性（contiguity）是指张量的元素在内存中是否连续存储。连续张量的元素在内存中按照特定的顺序（通常是行优先或列优先）连续排列，没有间隙。

```python
import torch

# 创建一个连续张量
contiguous_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Is contiguous: {contiguous_tensor.is_contiguous()}")
# 输出: Is contiguous: True

# 创建一个非连续张量（通过转置）
non_contiguous_tensor = contiguous_tensor.t()  # 转置
print(f"Transposed tensor: {non_contiguous_tensor}")
print(f"Transposed shape: {non_contiguous_tensor.shape}")
print(f"Transposed stride: {non_contiguous_tensor.stride()}")
print(f"Is contiguous: {non_contiguous_tensor.is_contiguous()}")
# 输出: Is contiguous: False

# 使张量连续
contiguous_version = non_contiguous_tensor.contiguous()
print(f"After contiguous(): {contiguous_version.is_contiguous()}")
# 输出: After contiguous(): True
```

连续性对性能有重要影响：
- 连续张量通常可以更高效地进行内存访问和计算
- 某些操作要求张量是连续的
- 非连续张量可以通过 `.contiguous()` 方法转换为连续张量，这会创建一个新的内存副本

### 张量的视图和重塑

张量的视图（view）和重塑（reshape）操作允许我们改变张量的形状而不改变其数据。

#### 张量的视图（View）

视图操作创建一个共享底层数据的新张量，但具有不同的形状。视图不会复制数据，而是提供了访问相同数据的不同方式。

```python
import torch

# 创建一个12元素的向量
vector = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# 创建一个3x4矩阵视图
matrix_view = vector.view(3, 4)
print(f"Vector: {vector}")
print(f"Matrix view: {matrix_view}")
# 输出:
# Vector: tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
# Matrix view: tensor([[ 1,  2,  3,  4],
#                      [ 5,  6,  7,  8],
#                      [ 9, 10, 11, 12]])

# 修改视图会影响原始张量
matrix_view[0, 0] = 99
print(f"Vector after modifying view: {vector}")
# 输出: Vector after modifying view: tensor([99,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

# 使用-1自动计算维度
auto_view = vector.view(4, -1)  # -1表示自动计算这个维度的大小
print(f"Auto-sized view: {auto_view}")
# 输出: Auto-sized view: tensor([[99,  2,  3],
#                               [ 4,  5,  6],
#                               [ 7,  8,  9],
#                               [10, 11, 12]])
```

视图操作的关键特性：
- 视图与原始张量共享相同的底层数据
- 修改视图会修改原始张量，反之亦然
- 视图操作要求张量是连续的，或者可以在不复制数据的情况下创建视图
- 使用 `-1` 作为维度大小可以自动计算该维度

#### 张量的重塑（Reshape）

重塑操作类似于视图，但它可以处理非连续张量。如果可能，`reshape` 会返回一个视图；如果不可能，它会返回一个新的连续张量。

```python
import torch

# 创建一个2x3矩阵
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 转置使其变为非连续
transposed = matrix.t()  # 形状为3x2
print(f"Transposed: {transposed}")
print(f"Is contiguous: {transposed.is_contiguous()}")
# 输出: Is contiguous: False

# 尝试创建视图（会失败）
try:
    view_attempt = transposed.view(6)
except RuntimeError as e:
    print(f"View error: {e}")
# 输出类似于: View error: view size is not compatible with input tensor's size and stride

# 使用reshape（会成功）
reshaped = transposed.reshape(6)
print(f"Reshaped: {reshaped}")
# 输出: Reshaped: tensor([1, 4, 2, 5, 3, 6])

# 检查是否创建了新的存储
print(f"Shares data with original: {reshaped.storage().data_ptr() == transposed.storage().data_ptr()}")
# 输出可能为True或False，取决于实现
```

重塑操作的关键特性：
- 如果可能，`reshape` 会返回一个视图（不复制数据）
- 如果不可能创建视图，`reshape` 会返回一个新的连续张量（复制数据）
- `reshape` 比 `view` 更灵活，但可能会导致数据复制

#### 展平和挤压操作

除了 `view` 和 `reshape` 外，还有一些常用的形状操作：

1. **展平（Flatten）**：
   将张量转换为一维向量。

   ```python
   import torch
   
   # 创建一个2x3矩阵
   matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
   
   # 展平
   flattened = matrix.flatten()
   print(f"Flattened: {flattened}")
   # 输出: Flattened: tensor([1, 2, 3, 4, 5, 6])
   
   # 也可以只展平特定维度
   partially_flattened = matrix.flatten(0, 0)  # 只展平第0维
   print(f"Partially flattened: {partially_flattened}")
   # 输出: Partially flattened: tensor([[1, 2, 3], [4, 5, 6]])
   ```

2. **挤压（Squeeze）**：
   移除大小为1的维度。

   ```python
   import torch
   
   # 创建一个形状为(1, 3, 1, 2)的张量
   tensor = torch.tensor([[[[1, 2]], [[3, 4]], [[5, 6]]]])
   print(f"Original shape: {tensor.shape}")
   # 输出: Original shape: torch.Size([1, 3, 1, 2])
   
   # 挤压所有大小为1的维度
   squeezed = tensor.squeeze()
   print(f"Squeezed shape: {squeezed.shape}")
   # 输出: Squeezed shape: torch.Size([3, 2])
   
   # 只挤压特定维度
   partially_squeezed = tensor.squeeze(0)  # 只挤压第0维
   print(f"Partially squeezed shape: {partially_squeezed.shape}")
   # 输出: Partially squeezed shape: torch.Size([3, 1, 2])
   ```

3. **扩展（Unsqueeze）**：
   添加大小为1的维度。

   ```python
   import torch
   
   # 创建一个形状为(3, 2)的张量
   tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
   print(f"Original shape: {tensor.shape}")
   # 输出: Original shape: torch.Size([3, 2])
   
   # 在第0维添加一个大小为1的维度
   unsqueezed_0 = tensor.unsqueeze(0)
   print(f"Unsqueezed at dim 0 shape: {unsqueezed_0.shape}")
   # 输出: Unsqueezed at dim 0 shape: torch.Size([1, 3, 2])
   
   # 在第1维添加一个大小为1的维度
   unsqueezed_1 = tensor.unsqueeze(1)
   print(f"Unsqueezed at dim 1 shape: {unsqueezed_1.shape}")
   # 输出: Unsqueezed at dim 1 shape: torch.Size([3, 1, 2])
   ```

### 张量的高级索引和切片

张量支持多种索引和切片操作，允许我们访问和修改特定的元素或子张量。

#### 基本索引

基本索引使用整数或整数列表访问特定元素。

```python
import torch

# 创建一个3x4矩阵
matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 访问单个元素
element = matrix[1, 2]
print(f"Element at [1, 2]: {element}")
# 输出: Element at [1, 2]: tensor(7)

# 访问一行
row = matrix[1]
print(f"Row 1: {row}")
# 输出: Row 1: tensor([5, 6, 7, 8])

# 访问一列
column = matrix[:, 2]
print(f"Column 2: {column}")
# 输出: Column 2: tensor([3, 7, 11])
```

#### 切片

切片使用 `start:end:step` 语法访问子张量。

```python
import torch

# 创建一个3x4矩阵
matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 切片行
rows_slice = matrix[0:2]  # 前两行
print(f"Rows 0:2: {rows_slice}")
# 输出: Rows 0:2: tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

# 切片列
cols_slice = matrix[:, 1:3]  # 所有行的第1列和第2列
print(f"Columns 1:3: {cols_slice}")
# 输出: Columns 1:3: tensor([[ 2,  3], [ 6,  7], [10, 11]])

# 使用步长
strided_slice = matrix[::2, ::2]  # 每隔一行和每隔一列
print(f"Strided slice: {strided_slice}")
# 输出: Strided slice: tensor([[ 1,  3], [ 9, 11]])
```

#### 高级索引

高级索引使用整数数组、布尔数组或它们的组合进行索引。

1. **整数数组索引**：

   ```python
   import torch
   
   # 创建一个3x4矩阵
   matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
   
   # 使用整数数组索引
   indices = torch.tensor([0, 2])
   selected_rows = matrix[indices]
   print(f"Selected rows: {selected_rows}")
   # 输出: Selected rows: tensor([[ 1,  2,  3,  4], [ 9, 10, 11, 12]])
   
   # 选择特定的(行,列)对
   row_indices = torch.tensor([0, 1, 2])
   col_indices = torch.tensor([1, 2, 0])
   elements = matrix[row_indices, col_indices]
   print(f"Selected elements: {elements}")
   # 输出: Selected elements: tensor([2, 7, 9])
   ```

2. **布尔索引**：

   ```python
   import torch
   
   # 创建一个3x4矩阵
   matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
   
   # 使用布尔掩码
   mask = matrix > 5
   print(f"Mask: {mask}")
   # 输出: Mask: tensor([[False, False, False, False],
   #                     [False,  True,  True,  True],
   #                     [ True,  True,  True,  True]])
   
   # 选择大于5的元素
   selected = matrix[mask]
   print(f"Elements > 5: {selected}")
   # 输出: Elements > 5: tensor([ 6,  7,  8,  9, 10, 11, 12])
   
   # 条件选择
   even_elements = matrix[matrix % 2 == 0]
   print(f"Even elements: {even_elements}")
   # 输出: Even elements: tensor([ 2,  4,  6,  8, 10, 12])
   ```

3. **组合索引**：

   ```python
   import torch
   
   # 创建一个3x4矩阵
   matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
   
   # 组合切片和整数索引
   combined = matrix[1:, [0, 2]]
   print(f"Combined indexing: {combined}")
   # 输出: Combined indexing: tensor([[ 5,  7], [ 9, 11]])
   
   # 组合整数索引和布尔索引
   row_indices = torch.tensor([0, 2])
   col_mask = torch.tensor([True, False, True, False])
   combined = matrix[row_indices][:, col_mask]
   print(f"Combined integer and boolean: {combined}")
   # 输出: Combined integer and boolean: tensor([[ 1,  3], [ 9, 11]])
   ```

### 张量的广播机制

广播（Broadcasting）是一种强大的机制，允许不同形状的张量进行元素级操作。广播规则自动扩展较小的张量以匹配较大的张量，而不需要实际复制数据。

#### 广播规则

PyTorch的广播规则与NumPy类似：

1. 从尾部维度开始比较两个张量的形状
2. 如果维度兼容（相等或其中一个为1），则继续
3. 如果维度不兼容，则引发错误

两个维度兼容的条件是：
- 它们相等，或
- 其中一个为1（在这种情况下，该维度会被"广播"以匹配另一个）

```python
import torch

# 示例1：标量和向量
scalar = torch.tensor(2)
vector = torch.tensor([1, 2, 3])
result = scalar + vector
print(f"Scalar + Vector: {result}")
# 输出: Scalar + Vector: tensor([3, 4, 5])
# 广播将标量2扩展为[2, 2, 2]

# 示例2：向量和矩阵
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
result = vector + matrix
print(f"Vector + Matrix: {result}")
# 输出: Vector + Matrix: tensor([[2, 4, 6], [5, 7, 9]])
# 广播将向量[1, 2, 3]扩展为[[1, 2, 3], [1, 2, 3]]

# 示例3：不兼容的形状
vector1 = torch.tensor([1, 2, 3])
vector2 = torch.tensor([1, 2])
try:
    result = vector1 + vector2
except RuntimeError as e:
    print(f"Broadcasting error: {e}")
# 输出类似于: Broadcasting error: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
```

#### 广播的应用

广播在深度学习中有许多应用：

1. **批量操作**：
   对批次中的每个样本应用相同的操作。

   ```python
   import torch
   
   # 批次数据：3个样本，每个样本4个特征
   batch = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
   
   # 每个特征的缩放因子
   scales = torch.tensor([0.1, 0.2, 0.3, 0.4])
   
   # 广播缩放
   scaled = batch * scales  # scales被广播到形状(3, 4)
   print(f"Scaled batch: {scaled}")
   # 输出: Scaled batch: tensor([[0.1, 0.4, 0.9, 1.6],
   #                            [0.5, 1.2, 2.1, 3.2],
   #                            [0.9, 2.0, 3.3, 4.8]])
   ```

2. **添加偏置**：
   向每个样本添加相同的偏置。

   ```python
   import torch
   
   # 批次数据：3个样本，每个样本4个特征
   batch = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
   
   # 偏置向量
   bias = torch.tensor([0.1, 0.2, 0.3, 0.4])
   
   # 添加偏置
   biased = batch + bias  # bias被广播到形状(3, 4)
   print(f"Biased batch: {biased}")
   # 输出: Biased batch: tensor([[ 1.1,  2.2,  3.3,  4.4],
   #                            [ 5.1,  6.2,  7.3,  8.4],
   #                            [ 9.1, 10.2, 11.3, 12.4]])
   ```

3. **掩码操作**：
   使用布尔掩码选择性地修改张量。

   ```python
   import torch
   
   # 创建一个3x4矩阵
   matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
   
   # 创建一个布尔掩码
   mask = matrix > 5
   
   # 使用掩码将大于5的元素设置为0
   masked = torch.where(mask, torch.zeros_like(matrix), matrix)
   print(f"Masked matrix: {masked}")
   # 输出: Masked matrix: tensor([[1, 2, 3, 4],
   #                             [5, 0, 0, 0],
   #                             [0, 0, 0, 0]])
   ```

### 张量的内存管理

有效的内存管理对于高性能深度学习至关重要，特别是在处理大型模型和数据集时。

#### 内存共享和复制

理解何时张量共享内存以及何时创建新的内存副本是很重要的：

```python
import torch

# 创建一个张量
original = torch.tensor([1, 2, 3, 4])

# 视图共享内存
view = original.view(2, 2)
print(f"View shares storage: {view.storage().data_ptr() == original.storage().data_ptr()}")
# 输出: View shares storage: True

# 克隆创建新的内存副本
clone = original.clone()
print(f"Clone shares storage: {clone.storage().data_ptr() == original.storage().data_ptr()}")
# 输出: Clone shares storage: False

# 某些操作会创建新的张量
add_result = original + 1
print(f"Add result shares storage: {add_result.storage().data_ptr() == original.storage().data_ptr()}")
# 输出: Add result shares storage: False
```

#### 就地操作

就地操作直接修改张量，而不是创建新的张量。这些操作通常以下划线后缀表示（如 `add_`）：

```python
import torch

# 创建一个张量
tensor = torch.tensor([1, 2, 3, 4])
original_ptr = tensor.storage().data_ptr()

# 非就地加法
result = tensor + 1
print(f"After non-inplace add: {tensor}")  # 原始张量不变
print(f"Result: {result}")
print(f"Original tensor unchanged: {tensor.storage().data_ptr() == original_ptr}")
# 输出: Original tensor unchanged: True

# 就地加法
tensor.add_(1)
print(f"After inplace add: {tensor}")  # 原始张量被修改
print(f"Original tensor still the same object: {tensor.storage().data_ptr() == original_ptr}")
# 输出: Original tensor still the same object: True
```

就地操作的优缺点：
- **优点**：减少内存使用和分配开销
- **缺点**：可能会破坏计算图，导致自动微分出错

#### 内存钉住和释放

在使用CUDA张量时，了解内存钉住（pinning）和释放是很重要的：

```python
import torch

# 创建一个CPU张量
cpu_tensor = torch.tensor([1, 2, 3, 4])

# 钉住内存（使其页锁定）
pinned_tensor = cpu_tensor.pin_memory()
print(f"Is pinned: {torch.cuda.is_pinned(pinned_tensor)}")
# 输出: Is pinned: True

# 创建一个CUDA张量
if torch.cuda.is_available():
    cuda_tensor = torch.tensor([1, 2, 3, 4], device="cuda")
    
    # 显式释放CUDA内存
    del cuda_tensor
    torch.cuda.empty_cache()
```

内存钉住的优缺点：
- **优点**：加速CPU到GPU的数据传输
- **缺点**：减少可用的系统内存，因为钉住的内存不能被操作系统分页

### 张量的设备管理

张量可以存储在不同的设备上，如CPU或GPU。设备管理对于高效的深度学习计算至关重要。

#### 设备分配

可以在创建张量时指定设备，或者稍后将张量移动到不同的设备：

```python
import torch

# 在CPU上创建张量
cpu_tensor = torch.tensor([1, 2, 3, 4])
print(f"CPU tensor device: {cpu_tensor.device}")
# 输出: CPU tensor device: cpu

# 在GPU上创建张量（如果可用）
if torch.cuda.is_available():
    cuda_tensor = torch.tensor([1, 2, 3, 4], device="cuda")
    print(f"CUDA tensor device: {cuda_tensor.device}")
    # 输出类似于: CUDA tensor device: cuda:0
    
    # 将CPU张量移动到GPU
    cuda_tensor2 = cpu_tensor.to("cuda")
    print(f"Moved tensor device: {cuda_tensor2.device}")
    # 输出类似于: Moved tensor device: cuda:0
    
    # 将GPU张量移动到CPU
    cpu_tensor2 = cuda_tensor.to("cpu")
    print(f"Moved back tensor device: {cpu_tensor2.device}")
    # 输出: Moved back tensor device: cpu
```

#### 设备间数据传输

设备间的数据传输可能很昂贵，应该尽量减少：

```python
import torch
import time

if torch.cuda.is_available():
    # 创建一个大张量
    large_tensor = torch.randn(10000, 10000)
    
    # 测量传输时间
    start = time.time()
    cuda_tensor = large_tensor.to("cuda")
    torch.cuda.synchronize()  # 确保GPU操作完成
    transfer_time = time.time() - start
    print(f"Transfer time: {transfer_time:.4f} seconds")
    
    # 测量传输回CPU的时间
    start = time.time()
    cpu_tensor = cuda_tensor.to("cpu")
    transfer_back_time = time.time() - start
    print(f"Transfer back time: {transfer_back_time:.4f} seconds")
```

减少设备间数据传输的策略：
- 尽早将数据移动到目标设备，并在那里保持
- 批量处理数据传输
- 使用内存钉住加速CPU到GPU的传输
- 使用异步传输重叠计算和数据传输

### 张量操作的性能考虑

在处理张量时，有几个性能因素需要考虑：

#### 连续性和内存访问模式

连续张量通常比非连续张量有更好的性能，因为它们具有更好的内存访问模式：

```python
import torch
import time

# 创建一个大矩阵
large_matrix = torch.randn(5000, 5000)

# 创建一个非连续视图（转置）
transposed = large_matrix.t()
print(f"Transposed is contiguous: {transposed.is_contiguous()}")
# 输出: Transposed is contiguous: False

# 测量连续和非连续张量的操作时间
start = time.time()
result1 = large_matrix.sum(dim=1)
contiguous_time = time.time() - start

start = time.time()
result2 = transposed.sum(dim=0)  # 等效操作，但在非连续张量上
non_contiguous_time = time.time() - start

print(f"Contiguous operation time: {contiguous_time:.6f} seconds")
print(f"Non-contiguous operation time: {non_contiguous_time:.6f} seconds")
print(f"Slowdown factor: {non_contiguous_time / contiguous_time:.2f}x")
```

#### 融合操作

融合操作将多个操作组合成一个，减少内存访问和同步开销：

```python
import torch

# 单独的操作
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

# 方法1：分步执行
a = x + y
b = a * 2
c = b.relu()

# 方法2：使用融合操作
c_fused = (x + y).mul(2).relu()

# 两种方法产生相同的结果，但融合操作通常更快
print(f"Results equal: {torch.allclose(c, c_fused)}")
# 输出: Results equal: True
```

在PyTorch中，可以使用JIT（Just-In-Time）编译和TorchScript进一步优化融合操作：

```python
import torch

# 定义一个简单的函数
def my_function(x, y):
    return (x + y).mul(2).relu()

# 创建输入
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

# 使用JIT编译
scripted_fn = torch.jit.script(my_function)

# 比较性能
import time

start = time.time()
result1 = my_function(x, y)
python_time = time.time() - start

start = time.time()
result2 = scripted_fn(x, y)
jit_time = time.time() - start

print(f"Python time: {python_time:.6f} seconds")
print(f"JIT time: {jit_time:.6f} seconds")
print(f"Speedup: {python_time / jit_time:.2f}x")
```

#### 批量处理

批量处理可以显著提高性能，特别是在GPU上：

```python
import torch
import time

# 创建数据
n_samples = 10000
x = torch.randn(n_samples, 100)
w = torch.randn(100, 100)

# 方法1：逐个处理样本
start = time.time()
results_individual = []
for i in range(n_samples):
    results_individual.append(torch.matmul(x[i], w))
results_individual = torch.stack(results_individual)
individual_time = time.time() - start

# 方法2：批量处理所有样本
start = time.time()
results_batch = torch.matmul(x, w)
batch_time = time.time() - start

print(f"Individual processing time: {individual_time:.6f} seconds")
print(f"Batch processing time: {batch_time:.6f} seconds")
print(f"Speedup: {individual_time / batch_time:.2f}x")
print(f"Results equal: {torch.allclose(results_individual, results_batch)}")
```

### 张量操作的实际应用

让我们看一些张量操作在深度学习中的实际应用：

#### 图像处理

图像通常表示为形状为 `(batch_size, channels, height, width)` 的4D张量：

```python
import torch
import torch.nn.functional as F

# 创建一个批次的图像（2张图像，3个通道，32x32像素）
images = torch.randn(2, 3, 32, 32)

# 1. 调整大小
resized = F.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
print(f"Resized shape: {resized.shape}")
# 输出: Resized shape: torch.Size([2, 3, 64, 64])

# 2. 标准化
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
normalized = (images - mean) / std

# 3. 数据增强 - 随机裁剪
batch, channels, height, width = images.shape
crop_height, crop_width = 24, 24
top = torch.randint(0, height - crop_height + 1, (batch,))
left = torch.randint(0, width - crop_width + 1, (batch,))

cropped = []
for i in range(batch):
    cropped.append(images[i, :, top[i]:top[i]+crop_height, left[i]:left[i]+crop_width])
cropped = torch.stack(cropped)
print(f"Cropped shape: {cropped.shape}")
# 输出: Cropped shape: torch.Size([2, 3, 24, 24])
```

#### 序列处理

序列数据（如文本）通常表示为形状为 `(batch_size, sequence_length, feature_dim)` 的3D张量：

```python
import torch

# 创建一个批次的序列（3个序列，每个序列长度为5，每个时间步10个特征）
sequences = torch.randn(3, 5, 10)

# 1. 掩码填充
# 假设序列的实际长度是[5, 3, 4]
lengths = torch.tensor([5, 3, 4])
mask = torch.arange(sequences.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
mask = mask.unsqueeze(-1).expand_as(sequences)
masked_sequences = sequences * mask.float()

# 2. 序列打包（用于RNN）
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
packed = pack_padded_sequence(sequences, lengths.cpu(), batch_first=True, enforce_sorted=False)

# 3. 注意力机制
query = torch.randn(3, 10)  # 查询向量
attention_scores = torch.bmm(sequences, query.unsqueeze(-1)).squeeze(-1)
attention_weights = torch.softmax(attention_scores, dim=1)
context = torch.bmm(attention_weights.unsqueeze(1), sequences).squeeze(1)
print(f"Context vector shape: {context.shape}")
# 输出: Context vector shape: torch.Size([3, 10])
```

#### 批量矩阵运算

批量矩阵运算在深度学习中非常常见：

```python
import torch

# 创建一批矩阵（32个样本，每个是10x20矩阵）
batch_A = torch.randn(32, 10, 20)
batch_B = torch.randn(32, 20, 30)

# 批量矩阵乘法
batch_C = torch.bmm(batch_A, batch_B)
print(f"Result shape: {batch_C.shape}")
# 输出: Result shape: torch.Size([32, 10, 30])

# 批量矩阵转置
batch_A_T = batch_A.transpose(1, 2)
print(f"Transposed shape: {batch_A_T.shape}")
# 输出: Transposed shape: torch.Size([32, 20, 10])

# 批量矩阵求逆
batch_square = torch.randn(32, 10, 10)
batch_inverse = torch.inverse(batch_square)
# 验证：A * A^(-1) ≈ I
identity_check = torch.bmm(batch_square, batch_inverse)
is_close_to_identity = torch.allclose(identity_check, torch.eye(10).unsqueeze(0).expand(32, 10, 10), atol=1e-5)
print(f"Is close to identity: {is_close_to_identity}")
```

### 总结

张量是深度学习中的基本数据结构，理解其形状、视图、步长和连续性对于高效的神经网络编程至关重要。本节探讨了张量的核心概念和操作，包括：

1. **张量的基本概念**：维度、形状和数据类型
2. **张量的存储和布局**：存储、步长和连续性
3. **张量的视图和重塑**：view、reshape、flatten和squeeze操作
4. **张量的高级索引和切片**：基本索引、切片和高级索引
5. **张量的广播机制**：自动扩展较小的张量以匹配较大的张量
6. **张量的内存管理**：内存共享、复制和就地操作
7. **张量的设备管理**：在CPU和GPU之间移动张量
8. **张量操作的性能考虑**：连续性、融合操作和批量处理
9. **张量操作的实际应用**：图像处理、序列处理和批量矩阵运算

通过掌握这些概念和技术，我们可以更有效地实现和优化深度学习模型，特别是在构建故事讲述AI系统时。张量操作的高效实现对于模型的训练和推理性能至关重要，直接影响用户体验和系统的可扩展性。
