# transformer_emotional_analysis

## 近期学习了transformer的一些构造，刚好未来可能会用到情感分类的功能，所以我这次制作一个详细地代码和readme解释，以供未来使用。

## 一、import板块

---

### 1. `import torch`
- **功能**：PyTorch 的**根引擎**。负责张量（Tensor）的数学运算、自动求导（Autograd）、GPU/CPU设备管理以及随机数种子。
- **标准用法**：所有数据（输入、标签、权重）都以 `torch.Tensor` 形式存在。
- ⚠️ **重要注意事项**：
  - **设备陷阱**：张量默认在CPU上。`torch.cuda.is_available()` 必须显式检查，否则在无GPU机器上强行`.cuda()`会报错。
  - **全局种子**：调用 `torch.manual_seed(42)` 只能固定CPU和主GPU，多卡训练还需额外设置 `torch.cuda.manual_seed_all(42)`，否则不同卡上的Dropout顺序不同，导致结果不可复现。

---

### 2. `import torch.nn as nn`
- **功能**：**神经网络工具箱**。包含所有预定义的层（`Linear`、`Embedding`、`LayerNorm`、`Dropout`）以及容器基类 `Module`。你写的所有模型类必须继承 `nn.Module`。
- **标准用法**：在 `__init__` 中实例化层（如 `self.linear = nn.Linear(...)`）。
- ⚠️ **重要注意事项**：
  - **默认初始化陷阱**：`nn.Linear` 默认使用 `Kaiming Uniform` 初始化（针对ReLU设计），但Transformer更适合 **Xavier（Glorot）初始化**。如果不手动重写初始化（如我在代码中做的 `nn.init.xavier_uniform_`），深层Transformer极容易出现梯度消失。
  - **`nn.ModuleList` vs Python列表**：如果你把子层放进普通Python列表（`[layer1, layer2]`），PyTorch不会注册这些参数，导致梯度无法更新！必须用 `nn.ModuleList` 或 `nn.Sequential` 包裹。

---

### 3. `import torch.nn.functional as F`
- **功能**：**函数式接口**。提供无状态（无参）的函数，如激活函数（`relu`, `gelu`）、池化（`max_pool2d`）、损失函数（`cross_entropy`）和注意力中的 `softmax`。
- **标准用法**：`x = F.gelu(x)` 或 `loss = F.cross_entropy(logits, labels)`。
- ⚠️ **重要注意事项**：
  - **与 `nn` 的区别**：`nn.ReLU()` 是有状态的层（包含 `inplace` 参数），而 `F.relu` 是无状态的纯函数。在 `forward` 中调用 `F.gelu` 更轻量，适合只计算不存储参数的操作。
  - **致命陷阱**：切勿在 `F.dropout` 中忘记设置 `training=self.training`！否则模型在 `eval()` 模式下依然会进行Dropout，导致验证集结果随机波动。而 `nn.Dropout` 会自动识别 `model.train()`/`eval()` 状态，更安全。

---

### 4. `import math`
- **功能**：Python 标准数学库。提供常数（`pi`, `e`）和基础函数（`sqrt`, `cos`, `exp`, `log`）。
- **标准用法**：**位置编码（Positional Encoding）** 中的 `math.log(10000.0)` 和 `math.sqrt(self.d_k)`。
- ⚠️ **重要注意事项**：
  - **张量与标量的混用**：`math.sqrt` 只能处理Python标量（`int`/`float`）。如果你传入 `torch.Tensor`，会报类型错误。务必确保传入的是整数（如 `self.d_k`）。
  - **性能**：`math` 函数在CPU上计算标量极快，但如果你要在GPU上做大量向量化运算，请改用 `torch.sqrt`，因为 `math` 无法利用GPU并行。

---

### 5. `import copy`
- **功能**：Python 标准库。用于对象的**深拷贝（`deepcopy`）**和浅拷贝（`copy`）。
- **标准用法**：在Transformer中，如果你想要6个参数**不共享**的编码器层，通常用 `copy.deepcopy(block)` 克隆。
- ⚠️ **重要注意事项**：
  - **深拷贝之坑**：`copy.deepcopy(model)` 会递归复制所有参数和张量，极其消耗内存且耗时。在训练大模型时几乎禁止使用。如果只是想共享架构配置，用 `copy.deepcopy` 拷贝 `Config` 类即可。
  - **替代方案**：更推荐用 `nn.ModuleList([TransformerBlock() for _ in range(n_layers)])` 生成新实例，而不是深拷贝已有实例。

---

### 6. `from torch.optim import AdamW`
- **功能**：引入 **AdamW 优化器**。这是Adam算法的修正版，将**权重衰减（Weight Decay）**与损失函数解耦，实现了真正的L2正则化。
- **标准用法**：`optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)`。
- ⚠️ **重要注意事项（重点！）**：
  - **Adam vs AdamW**：老版Adam的 `weight_decay` 会直接加在梯度上，随学习率缩放，导致大学习率时正则失效。**AdamW 永远比 Adam 泛化好**，NLP领域绝不用Adam。
  - **参数分组**：可以传入字典列表，对 `bias` 和 `LayerNorm` 层的参数**禁用权重衰减**（因为这些层不需要正则化），标准做法是 `[{'params': no_decay}, {'params': decay, 'weight_decay': 0.01}]`。不这样做，小模型容易欠拟合。

---

### 7. `from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR`
- **功能**：**学习率调度器**。
  - `LambdaLR`：允许你自定义任意函数（`lambda step: ...`）来缩放学习率。
  - `CosineAnnealingLR`：按余弦周期衰减学习率（从当前值降到接近0）。
- **标准用法**：本代码中我用了 `LambdaLR` 手写“预热+余弦衰减”组合。
- ⚠️ **重要注意事项**：
  - **冗余导入**：细心观察，我的代码最终**没有使用** `CosineAnnealingLR`（因为用 `LambdaLR + math.cos` 替代了）。留着这个导入是为了方便你切换策略。注意：`CosineAnnealingLR` 默认**重启周期为总epoch数**，如果你跑多轮循环，它会直接降到0不再回升，不适合持续训练。
  - **调用时机**：必须**在每个batch后**调用 `scheduler.step()`（而不是每个epoch后），否则“预热”的步数计算会错乱。

---

### 8. `from torch.cuda.amp import autocast, GradScaler`
- **功能**：PyTorch 的**自动混合精度（AMP）**工具箱。彻底解决显存不足和训练龟速问题。
  - `autocast`：上下文管理器，在 `with` 块内自动将计算图的前向传播转为 `float16`。
  - `GradScaler`：缩放损失值，防止 `float16` 梯度下溢（变成0）。
- **标准用法**：前向传播包裹 `with autocast():`，反向时先 `scaler.scale(loss).backward()`，再 `scaler.step(optimizer)`。
- ⚠️ **重要注意事项（踩坑重灾区！）**：
  - **只在GPU生效**：CPU上调用 `autocast` 毫无作用且可能报错，必须有 `if torch.cuda.is_available()` 守卫。
  - **禁用操作**：`autocast` 块内**禁止**使用 `.item()` 提取标量，也禁止 `numpy()` 转换，否则会触发隐式同步，大幅拖慢速度。
  - **梯度裁剪**：必须先调用 `scaler.unscale_(optimizer)` 再裁剪梯度，否则梯度是缩放的，裁剪阈值会错乱。

---

### 9. `from torch.utils.data import DataLoader, Dataset`
- **功能**：**数据管道的基石**。
  - `Dataset`：抽象类，你需要重写 `__len__` 和 `__getitem__` 来定义数据读取逻辑。
  - `DataLoader`：迭代器，负责**批处理（Batching）**、**打乱（Shuffling）**、**多进程加载（num_workers）**。
- **标准用法**：`loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)`。
- ⚠️ **重要注意事项（系统级陷阱）**：
  - **`num_workers` 与内存**：Linux 下 `fork()` 会复制主进程内存，如果主进程已加载大数据集（如全部文本），开 `num_workers=8` 会瞬间占满内存。建议在 `__init__` 里只存文件路径，在 `__getitem__` 里动态读数据（懒加载）。
  - **`pin_memory=True`**：设置后会锁定内存页，加速CPU到GPU的传输（约提升20%），必须搭配 `non_blocking=True` 在 `to(device)` 中使用。

---

### 10. `import numpy as np`
- **功能**：科学计算库。处理多维数组、切片、统计。
- **标准用法**：数据预处理（读取CSV、归一化）、生成掩码、或与 `torch` 互转（`torch.from_numpy(np_array)`）。
- ⚠️ **重要注意事项**：
  - **类型转换陷阱**：Numpy 默认 `float64`，而 PyTorch 模型默认 `float32`。直接 `torch.from_numpy(np.float64)` 转张量会导致模型前向计算时类型不匹配报错！必须显式转：`torch.from_numpy(np_array.astype(np.float32))`。
  - **随机种子隔离**：Numpy 有自己的随机状态（`np.random.seed`），必须和 `torch.manual_seed` 同时设置，否则数据增强会不一样。

---
