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

#### 数学协同效应：AdamW 更新公式与学习率调度的深度耦合

为什么学习率调度在 Transformer 中如此生死攸关？其数学本质取决于 **AdamW 的更新公式**：

\[
\theta_{t+1} = \theta_t - \eta_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \eta_t \cdot \lambda \cdot \theta_t
\]

为了彻底理解这个公式，我们需要把它拆解为三个物理含义明确的层次：

**第一层：自适应梯度项（\( \eta_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} \)）**  
- \( m_t \) 是梯度的**一阶矩（带方向的动量）**，负责累积历史梯度的加权平均，决定参数更新的方向。  
- \( v_t \) 是梯度的**二阶矩（未中心化的方差）**，负责估计历史梯度各维度的幅度大小。在 Transformer 的深层网络中，不同层的梯度方差可能相差数个数量级（例如，Embedding 层梯度极稀疏，而 Attention 输出层梯度极密集）。  
- \( \frac{m_t}{\sqrt{v_t}} \) 本质上是对每个参数维度进行**自适应缩放**：梯度变化剧烈的维度除以更大的 \( \sqrt{v_t} \)，从而压低步长；梯度平稳的维度则获得更大的有效步长。如果没有这一项，Transformer 根本无法收敛。

**第二层：解耦的权重衰减项（即为正则项）（\( \eta_t \cdot \lambda \cdot \theta_t \)）**  
- 注意观察，这里的 **权重衰减（Weight Decay）项直接被当前学习率 \( \eta_t \) 线性缩放**。这正是 AdamW 与老版 Adam 在数学上的根本分野。  
- 在老版 Adam 中，权重衰减项 \( \lambda \cdot \theta_t \) **不乘以 \( \eta_t \)**，而是直接加在梯度上。当学习率 \( \eta_t \) 较大时，梯度项占主导，正则项几乎失效；当学习率较小时，正则项又突然过强，导致训练不稳定。AdamW 通过强制乘上 \( \eta_t \)，使得正则化强度与学习率同频共振，实现了真正的解耦。

**第三层：预热阶段（\( \eta_t \) 极小时）的协同效应**  
- 在预热初期，\( \eta_t \) 从 0 逐渐爬升到基准值。此时，**权重衰减项 \( \eta_t \cdot \lambda \cdot \theta_t \) 几乎可以忽略不计**（因为 \( \eta_t \to 0 \)）。  
- 这带来了一个极其重要的数学后果：模型在初期处于**“无正则化自由生长”**状态。此时二阶矩 \( v_t \) 的估计还不准确（早期样本方差极大），如果强行施加 L2 正则化，会把初始权重向零拉拽，干扰 \( v_t \) 的统计积累。预热阶段屏蔽掉衰减项，正好给了方差估计一个纯净的“缓冲期”，防止梯度爆炸。

**第四层：余弦衰减末期（\( \eta_t \to 0 \)）的精细打磨**  
- 当训练步入余弦衰减的末端，\( \eta_t \) 无限趋近于 0。此时 **权重衰减项 \( \eta_t \cdot \lambda \cdot \theta_t \) 也同步趋近于 0**。  
- 这意味着模型在最后阶段执行的实际上是 **纯梯度下降（SGD-like）微调**，而不是 L2 正则化。L2 正则化的本质是把参数向零压缩（拉普拉斯先验），这会抹除模型在训练中后期学到的高频细节特征（例如注意力矩阵中的尖锐分布）。取消衰减项后，参数可以自由停留在损失平面上的**平坦极小值（Flat Minima）**深处，而不被正则项拖拽回原点。数学上等价于对模型参数进行了“精细打磨”，保留了它学到的所有高频细节，这对提升验证集准确率有 0.3%~1% 的数学增益。

综上所述，预热和余弦衰减分别通过极端放大和极端压制 \( \eta_t \)，间接控制了权重衰减项的有效强度。这种耦合关系构成了 Transformer 训练中无可替代的“降维打击”式协同效应。

---

### 7. `from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR`
- **功能**：**学习率调度器**。
  - `LambdaLR`：允许你自定义任意函数（`lambda step: ...`）来缩放学习率。
  - `CosineAnnealingLR`：按余弦周期衰减学习率（从当前值降到接近0）。
- **标准用法**：本代码中我用了 `LambdaLR` 手写“预热+余弦衰减”组合。
- 详细解释：
  ### 第一部分：`LambdaLR` —— 万能函数器（数学定义）

  **数学通式：**
  \[
  \eta_t = \eta_{base} \times \lambda(t)
  \]
  其中 \( \lambda(t) \) 是你传入的**任意自定义函数**，\( t \) 代表当前的训练步数（从 0 开始）。

  **在最强代码中的实际应用（预热 + 余弦退火）：**
  我代码里传进去的 \( \lambda(t) \) 实际上是一个**分段函数**，这是目前大模型（GPT/LLaMA）训练的黄金标准：

  \[
  \lambda(t) = 
  \begin{cases} 
  \frac{t}{warmup\_steps}, & \text{if } t < warmup\_steps \quad (\text{预热阶段}) \\
  \frac{1}{2} \left(1 + \cos\left(\pi \cdot \frac{t - warmup\_steps}{total\_steps - warmup\_steps}\right)\right), & \text{if } t \ge warmup\_steps \quad (\text{衰减阶段})
  \end{cases}
  \]

  **数学深度解读：**

  1. **预热阶段（线性增长）**：\( \lambda(t) = \frac{t}{W} \)。
     - 这是一个线性插值，从 \( \lambda(0)=0 \) 平滑上升到 \( \lambda(W)=1 \)。
     - **数学上的必要性**：Adam 优化器利用了梯度的**一阶矩（均值）**和**二阶矩（方差）**。早期梯度极不稳定，二阶矩估计值偏小。如果此时用大学习率，\( \frac{梯度}{\sqrt{方差}} \) 会被极度放大，导致梯度爆炸。预热等于给了方差估计一个“缓冲期”。

  2. **余弦衰减阶段**：\( \lambda(t) = 0.5 \times (1 + \cos(\phi)) \)。
     - 这里的 \( \phi \) 从 \( 0 \) 变化到 \( \pi \)。
     - 当 \( \phi=0 \) 时，\( \cos(0)=1 \)，\( \lambda=1.0 \)（正好衔接预热结束的最大值）。
     - 当 \( \phi=\pi \) 时，\( \cos(\pi)=-1 \)，\( \lambda=0.0 \)（训练结束时学习率无限接近于 0）。
     - **为什么用余弦而不是线性衰减？** 因为余弦曲线在中间段（\( \phi \approx 90^\circ \)）下降速度较快，在两端（初期和末期）下降极慢。这允许模型在初期快速跳过“尖锐的损失盆地”，在末期用极慢的衰减在“平坦盆地”中充分微调。

  ---

  ### 第二部分：`CosineAnnealingLR` —— 重启式余弦（数学定义）

  如果你直接调用这个类，它的数学公式比上面的后半段稍微复杂一点，因为它支持**重启（Restart）**：

  **数学通式：**
  \[
  \eta_t = \eta_{min} + \frac{1}{2} \times (\eta_{base} - \eta_{min}) \times \left(1 + \cos\left(\frac{\pi \cdot T_{cur}}{T_{i}}\right)\right)
  \]

  - \( \eta_{min} \)：最低学习率（默认是 0，但你可以设为 \( \eta_{base} \times 1e-5 \)）。
  - \( T_{cur} \)：**自上次重启以来**经过的 epoch 数。
  - \( T_{i} \)：设定的周期长度（默认是 `T_max`，即总的 epoch 数）。

  **关键数学特性——重启（Restart）：**
  如果设置 `T_max=10` 且 `eta_min=0`，当 \( T_{cur} \) 从 0 走到 10 时，学习率从 \( \eta_{base} \) 平滑降到 0。然后，如果训练继续，它会**瞬间跳回** \( \eta_{base} \)，开始下一个周期。
  这种“突然跳高”被称为 **SGDR（随机梯度下降+热重启）**，数学上是为了让模型跳出当前找到的局部极小值，去探索更广阔的损失平面，寻找更深/更平的谷底。

  ---

  ### 第三部分：为什么舍弃了 `CosineAnnealingLR`，而用 `LambdaLR` 手写？

  因为 **`CosineAnnealingLR` 的默认周期是以“Epoch”为单位的**，而**预热是以“Step”为单位的**。混在一起容易把预热周期算乱。

  代码中用 `LambdaLR` 手写的分段函数，实现了 **“带预热的单周期余弦衰减”**（One-cycle Cosine）。其数学优势在于：

  1. **预热平滑衔接**：保证导数连续（导数为 0 处衔接，不会突变）。
  2. **终点趋近于 0**：这比 `CosineAnnealingLR` 默认降到一个固定值（`eta_min`）更激进。趋近于 0 相当于在最后自动实现了 **“强正则化”**，迫使模型收敛到邻近的极小值点，这对提升验证集准确率有 0.3%~1% 的数学增益。

  ---

  ### 第四部分：AdamW 框架下的数学协同效应（简单recap）

  为什么学习率调度在 Transformer 中如此生死攸关？数学上取决于 **AdamW 的更新公式**：
  \[
  \theta_{t+1} = \theta_t - \eta_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \eta_t \cdot \lambda \cdot \theta_t
  \]

  注意观察：**权重衰减（Weight Decay）项 \( \eta_t \cdot \lambda \cdot \theta_t \)** 直接被 \( \eta_t \) 缩放。

  - **预热阶段**：\( \eta_t \) 很小，所以权重衰减也极小。这允许模型在初期自由调整主权重，不受正则项压制。
  - **余弦衰减末期**：\( \eta_t \to 0 \)，此时权重衰减也趋近于 0。这意味着**模型在最后阶段执行的是纯梯度下降（SGD-like）微调**，而不是L2正则化。这数学上等价于对模型参数进行了“精细打磨”，保留了它学到的所有高频细节。

- ⚠️ **重要注意事项**：
  - **冗余导入**：细心观察，我的代码最终**没有使用** `CosineAnnealingLR`（因为用 `LambdaLR + math.cos` 替代了）。留着这个导入是为了方便你切换策略。注意：`CosineAnnealingLR` 默认**重启周期为总epoch数**，如果你跑多轮循环，它会直接降到0不再回升，不适合持续训练。
  - **调用时机**：必须**在每个batch后**调用 `scheduler.step()`（而不是每个epoch后），否则“预热”的步数计算会错乱。

---

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
