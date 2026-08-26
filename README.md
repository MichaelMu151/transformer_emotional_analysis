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

$$
\theta_{t+1} = \theta_t - \eta_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \eta_t \cdot \lambda \cdot \theta_t
$$

为了彻底理解这个公式，我们需要把它拆解为三个物理含义明确的层次：

**第一层：自适应梯度项（$ \eta_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} $）**  
- $ m_t $ 是梯度的**一阶矩（带方向的动量）**，负责累积历史梯度的加权平均，决定参数更新的方向。  
- $ v_t $ 是梯度的**二阶矩（未中心化的方差）**，负责估计历史梯度各维度的幅度大小。在 Transformer 的深层网络中，不同层的梯度方差可能相差数个数量级（例如，Embedding 层梯度极稀疏，而 Attention 输出层梯度极密集）。  
- $ \frac{m_t}{\sqrt{v_t}} $ 本质上是对每个参数维度进行**自适应缩放**：梯度变化剧烈的维度除以更大的 $ \sqrt{v_t} $，从而压低步长；梯度平稳的维度则获得更大的有效步长。如果没有这一项，Transformer 根本无法收敛。

**第二层：解耦的权重衰减项（即为正则项）（$ \eta_t \cdot \lambda \cdot \theta_t $）**  
- 注意观察，这里的 **权重衰减（Weight Decay）项直接被当前学习率 $ \eta_t $ 线性缩放**。这正是 AdamW 与老版 Adam 在数学上的根本分野。  
- 在老版 Adam 中，权重衰减项 $ \lambda \cdot \theta_t $ **不乘以 $ \eta_t $**，而是直接加在梯度上。当学习率 $ \eta_t $ 较大时，梯度项占主导，正则项几乎失效；当学习率较小时，正则项又突然过强，导致训练不稳定。AdamW 通过强制乘上 $ \eta_t $，使得正则化强度与学习率同频共振，实现了真正的解耦。

**第三层：预热阶段（$ \eta_t $ 极小时）的协同效应**  
- 在预热初期，$ \eta_t $ 从 0 逐渐爬升到基准值。此时，**权重衰减项 $ \eta_t \cdot \lambda \cdot \theta_t $ 几乎可以忽略不计**（因为 $ \eta_t \to 0 $）。  
- 这带来了一个极其重要的数学后果：模型在初期处于**“无正则化自由生长”**状态。此时二阶矩 $ v_t $ 的估计还不准确（早期样本方差极大），如果强行施加 L2 正则化，会把初始权重向零拉拽，干扰 $ v_t $ 的统计积累。预热阶段屏蔽掉衰减项，正好给了方差估计一个纯净的“缓冲期”，防止梯度爆炸。

**第四层：余弦衰减末期（$ \eta_t \to 0 $）的精细打磨**  
- 当训练步入余弦衰减的末端，$ \eta_t $ 无限趋近于 0。此时 **权重衰减项 $ \eta_t \cdot \lambda \cdot \theta_t $ 也同步趋近于 0**。  
- 这意味着模型在最后阶段执行的实际上是 **纯梯度下降（SGD-like）微调**，而不是 L2 正则化。L2 正则化的本质是把参数向零压缩（拉普拉斯先验），这会抹除模型在训练中后期学到的高频细节特征（例如注意力矩阵中的尖锐分布）。取消衰减项后，参数可以自由停留在损失平面上的**平坦极小值（Flat Minima）**深处，而不被正则项拖拽回原点。数学上等价于对模型参数进行了“精细打磨”，保留了它学到的所有高频细节，这对提升验证集准确率有 0.3%~1% 的数学增益。

综上所述，预热和余弦衰减分别通过极端放大和极端压制 $ \eta_t $，间接控制了权重衰减项的有效强度。这种耦合关系构成了 Transformer 训练中无可替代的“降维打击”式协同效应。

---

### 7. `from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR`
- **功能**：**学习率调度器**。
  - `LambdaLR`：允许你自定义任意函数（`lambda step: ...`）来缩放学习率。
  - `CosineAnnealingLR`：按余弦周期衰减学习率（从当前值降到接近0）。
- **标准用法**：本代码中我用了 `LambdaLR` 手写“预热+余弦衰减”组合。
- 详细解释：
  ### 第一部分：`LambdaLR` —— 万能函数器（数学定义）

  **数学通式：**
  $$
  \eta_t = \eta_{base} \times \lambda(t)
  $$
  其中 $ \lambda(t) $ 是你传入的**任意自定义函数**，$ t $ 代表当前的训练步数（从 0 开始）。

  **在最强代码中的实际应用（预热 + 余弦退火）：**
  我代码里传进去的 $ \lambda(t) $ 实际上是一个**分段函数**，这是目前大模型（GPT/LLaMA）训练的黄金标准：

  $$
  \lambda(t) = 
  \begin{cases} 
  \frac{t}{warmup\_steps}, & \text{if } t < warmup\_steps \quad (\text{预热阶段}) \\
  \frac{1}{2} \left(1 + \cos\left(\pi \cdot \frac{t - warmup\_steps}{total\_steps - warmup\_steps}\right)\right), & \text{if } t \ge warmup\_steps \quad (\text{衰减阶段})
  \end{cases}
  $$

  **数学深度解读：**

  1. **预热阶段（线性增长）**：$ \lambda(t) = \frac{t}{W} $。
     - 这是一个线性插值，从 $ \lambda(0)=0 $ 平滑上升到 $ \lambda(W)=1 $。
     - **数学上的必要性**：Adam 优化器利用了梯度的**一阶矩（均值）**和**二阶矩（方差）**。早期梯度极不稳定，二阶矩估计值偏小。如果此时用大学习率，$ \frac{梯度}{\sqrt{方差}} $ 会被极度放大，导致梯度爆炸。预热等于给了方差估计一个“缓冲期”。

  2. **余弦衰减阶段**：$ \lambda(t) = 0.5 \times (1 + \cos(\phi)) $。
     - 这里的 $ \phi $ 从 $ 0 $ 变化到 $ \pi $。
     - 当 $ \phi=0 $ 时，$ \cos(0)=1 $，$ \lambda=1.0 $（正好衔接预热结束的最大值）。
     - 当 $ \phi=\pi $ 时，$ \cos(\pi)=-1 $，$ \lambda=0.0 $（训练结束时学习率无限接近于 0）。
     - **为什么用余弦而不是线性衰减？** 因为余弦曲线在中间段（$ \phi \approx 90^\circ $）下降速度较快，在两端（初期和末期）下降极慢。这允许模型在初期快速跳过“尖锐的损失盆地”，在末期用极慢的衰减在“平坦盆地”中充分微调。

  ---

  ### 第二部分：`CosineAnnealingLR` —— 重启式余弦（数学定义）

  如果你直接调用这个类，它的数学公式比上面的后半段稍微复杂一点，因为它支持**重启（Restart）**：

  **数学通式：**
  $$
  \eta_t = \eta_{min} + \frac{1}{2} \times (\eta_{base} - \eta_{min}) \times \left(1 + \cos\left(\frac{\pi \cdot T_{cur}}{T_{i}}\right)\right)
  $$

  - $ \eta_{min} $：最低学习率（默认是 0，但你可以设为 $ \eta_{base} \times 1e-5 $）。
  - $ T_{cur} $：**自上次重启以来**经过的 epoch 数。
  - $ T_{i} $：设定的周期长度（默认是 `T_max`，即总的 epoch 数）。

  **关键数学特性——重启（Restart）：**
  如果设置 `T_max=10` 且 `eta_min=0`，当 $ T_{cur} $ 从 0 走到 10 时，学习率从 $ \eta_{base} $ 平滑降到 0。然后，如果训练继续，它会**瞬间跳回** $ \eta_{base} $，开始下一个周期。
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
  $$
  \theta_{t+1} = \theta_t - \eta_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \eta_t \cdot \lambda \cdot \theta_t
  $$

  注意观察：**权重衰减（Weight Decay）项 $ \eta_t \cdot \lambda \cdot \theta_t $** 直接被 $ \eta_t $ 缩放。

  - **预热阶段**：$ \eta_t $ 很小，所以权重衰减也极小。这允许模型在初期自由调整主权重，不受正则项压制。
  - **余弦衰减末期**：$ \eta_t \to 0 $，此时权重衰减也趋近于 0。这意味着**模型在最后阶段执行的是纯梯度下降（SGD-like）微调**，而不是L2正则化。这数学上等价于对模型参数进行了“精细打磨”，保留了它学到的所有高频细节。

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

## 二、超参数配置类（`class Config`）

---

### 1. `vocab_size`（词表维度）
- **物理直觉**：这是输入空间的“宇宙大小”。它决定了模型能辨识多少种不同的“基本粒子”（Token）。
- **数学本质**：独热编码（One-hot）的维度。嵌入矩阵 $ \mathbf{E} \in \mathbb{R}^{vocab\_size \times d\_model} $ 的行数。
- **信息论视角**：它定义了输入信道的最大香农熵。如果真实词表有 50,000 个词，而你只设了 10,000，那么 $ 10,000 $ 个类别之外的所有 Token 都会被强行映射为 `<UNK>`（未知词），导致信息丢失（困惑度 PPL 永久性升高）。
- **工程铁律**：必须保证 `max(input_ids) < vocab_size`。否则 PyTorch 在进行 `Embedding` 查表时会抛出 `IndexError`，直接崩溃。

---

### 2. `d_model`（隐层维度 / 模型宽度）
- **物理直觉**：这是模型内部处理信息时的“思考带宽”。类比于人的工作记忆容量——带宽越大，能同时关联的信息片段就越多。
- **数学本质**：它是所有 Transformer 层之间传递的张量的**最后一维大小**。决定了权重矩阵 $ \mathbf{W} \in \mathbb{R}^{d\_model \times d\_model} $ 的规模。
- **容量定律（Scaling Law）**：模型的参数量近似为 $ 12 \cdot n\_layers \cdot d\_model^2 $（忽略 FFN）。这意味着：
  - $ d\_model $ 从 512 提升到 768，参数量增长为 $ (768/512)^2 \approx 2.25 $ 倍（翻倍还多）。
- **训练动力学**：$ d\_model $ 越大，注意力分数 $ \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d\_k}} $ 的方差越稳定。但过大的 $ d\_model $ 会导致 Adam 优化器的二阶矩估计（$ v_t $）占用巨量显存（$ O(d\_model^2) $）。

---

### 3. `d_ff`（前馈网络维度）
- **物理直觉**：Transformer 的 FFN 被广泛认为是一个**“键值记忆网络”**（Key-Value Memory）。$ d\_ff $ 就是记忆存储的“条目数”。
- **数学本质**：两层线性变换的中间维度。全连接层 1 将 $ d\_model $ 升维到 $ d\_ff $，激活函数（GeLU）引入非线性，全连接层 2 将 $ d\_ff $ 降维回 $ d\_model $。
- **经验公式**：标准配置 $ d\_ff = 4 \times d\_model $。为什么是 4 倍？Transformer 原始论文通过消融实验发现，当 $ d\_ff $ 小于 $ 2 \times d\_model $ 时，模型表达能力骤降；大于 $ 4 \times d\_model $ 时，收益边际递减，且计算量（FLOPs）直线上升。
- **物理学类比**：类似于“压缩感知”。先扩张到高维空间寻找线性可分的超平面（Cover 定理），再压缩回原空间。$ d\_ff $ 决定了高维空间的曲率半径。

---

### 4. `n_heads`（注意力头数）
- **物理直觉**：人类在理解一句话时，会同时关注“语法依存”和“语义相似”——不同头就是不同的“观察视角”。
- **数学本质**：多头注意力是将 $ d\_model $ 拆分成 $ n\_heads $ 个并行的子空间。每个头的维度 $ d\_k = d\_model / n\_heads $。
- **特征分解**：数学上等价于对注意力矩阵进行**低秩分解**。$ n\_heads $ 越大，模型对位置和通道的建模越精细，但单头维度变小后，$ \mathbf{Q}\mathbf{K}^T $ 点积的结果会过于“稀疏”，导致 Softmax 退化为近似 One-hot（梯度极度稀疏，难以训练）。
- **约束条件**：必须满足 $ d\_model \bmod n\_heads = 0 $。否则无法进行维度均分。

---

### 5. `n_layers`（编码器层数 / 深度）
- **物理直觉**：这是模型进行“抽象层次”推理的深度。第一层看词性，中间层看短语，最后一层看全局意图。
- **数学本质**：函数复合的深度。$ f(x) = f_L(f_{L-1}(...f_1(x))) $。每一层都在对输入流形进行非线性扭曲。
- **梯度传播危机**：随着 $ n\_layers $ 增加，梯度消失/爆炸风险呈指数级上升。这就是为什么现代大模型必须使用 **Pre-LN（前置层归一化）** 而非 Post-LN。Pre-LN 保证了梯度可以在残差连接中无阻碍地直接回传（类似于 ResNet 的恒等映射）。
- **有效感受野**：堆叠 6 层时，理论上每个 Token 能感知到 $ 2^6 = 64 $ 个 Token 的上下文；堆叠 12 层时，可达 $ 2^{12} = 4096 $ 个 Token，极大增强长程依赖捕获能力。

---

### 6. `dropout`（随机失活比率）
- **物理直觉**：这是一种“贝叶斯正则化”。相当于在每次前向传播时，随机让一部分神经元“静默罢工”，强迫模型不依赖任何单一路径。
- **数学本质**：伯努利随机掩码 $ \mathbf{M} \sim \text{Bernoulli}(1-p) $，输出 $ \mathbf{Y} = \frac{1}{1-p} (\mathbf{M} \odot \mathbf{X}) $（缩放保持期望值不变）。
- **模型平均效应**：在测试时，Dropout 关闭，相当于同时运行了指数级数量（$ 2^N $）的子网络并取平均。这是一种轻量级的集成学习。
- **调参边界**：当训练数据量极大（>100万）时，Dropout 可以设得很低（0.05）甚至为 0，因为数据本身已提供了充足的正则化；当数据量少（<1万）时，Dropout=0.3 能有效防止训练集过拟合（验证集与训练集 Loss 差距过大）。

---

### 7. `label_smoothing`（标签平滑系数）
- **物理直觉**：防止模型“过度自信”。真实标签是 1，但模型没必要输出 1.0，输出 0.85 并保留 0.15 的不确定性，反而能让决策边界更宽泛。
- **数学公式推导**：
  设硬标签为独热向量 $ y_k = 1 $（正确类），平滑后的标签为：
  $$
  y'_k = (1 - \alpha) \cdot 1_{k=target} + \frac{\alpha}{K}
  $$
  其中 $ K $ 是类别数，$ \alpha $ 是平滑系数。

  对应的交叉熵损失变为：
  $$
  \mathcal{L} = (1-\alpha) \cdot \mathcal{L}_{CE} + \alpha \cdot \mathcal{L}_{KL}
  $$
  这相当于强迫模型在正确类与均匀分布之间寻求平衡，直接约束了模型权重的 Lipschitz 常数，使得对抗样本攻击更难成功。

- **工程注意**：PyTorch 的 `CrossEntropyLoss` 内置了 `label_smoothing` 参数，无需手动构造软标签。

---

### 8. `lr` & `warmup_steps`（学习率与预热）
- **物理直觉**：预热相当于飞机起飞时的“滑跑加速”。初始阶段，Adam 的梯度方差估计（二阶矩）严重偏小，若直接满油门（大学习率），梯度方向会被极度放大，直接失控。
- **数学本质**（Adam 方差修正）：
  Adam 的参数更新公式为 $ \Delta \theta = -\eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} $。在初始阶段，$ v_t $（梯度的平方滑动平均）非常小（接近于 0），导致 $ \frac{1}{\sqrt{v_t}} $ 极大。因此，学习率必须乘以一个小于 1 的缩放因子（$ \lambda(t) $）来抵消这种放大效应。
- **缩放因子的数学形式**：
  $$
  \lambda(t) = \frac{t}{W} \quad (t < W)
  $$
  当 $ t = W $ 时，$ \lambda=1 $，此时模型已预热完毕，$ v_t $ 也累积了足够的统计量，可以安全使用满额学习率 $ 1e-4 $。
- **劣质调参的报应**：如果不预热，前 100 步 Loss 会直接变成 `NaN`（浮点数溢出），因为梯度更新步长 $ \Delta \theta $ 超过了权重的稳定边界。

---

### 9. `max_seq_len`（最大序列长度）
- **物理直觉**：模型的“视窗大小”。决定了模型能同时看到多长的上下文。
- **数学复杂度（致命陷阱）**：Transformer 自注意力机制的时间和空间复杂度均为 **$ O(L^2) $**（$ L $ 为序列长度）。
  - 当 $ L=128 $ 时，注意力矩阵元素数为 $ 128^2 = 16,384 $。
  - 当 $ L=512 $ 时，元素数为 $ 512^2 = 262,144 $。
  - **结论**：$ L $ 翻 4 倍，显存占用和计算耗时翻 16 倍。这也是为什么长序列必须引入 Sparse Attention（稀疏注意力）或 FlashAttention 的原因。

---

### 10. `device`（硬件设备）
- **物理直觉**：程序运行的物理载体（硅基芯片）。
- **CUDA 本质**：`torch.device` 告诉 PyTorch 将张量分配在 CPU 的 DRAM 还是 GPU 的 VRAM 上。GPU 拥有数千个 CUDA 核心，适合大规模矩阵乘法（GEMM）的并行加速；CPU 适合逻辑控制和数据预处理。
- **生产级警告**：务必使用 `torch.cuda.is_available()` 做动态检测。若直接将模型写到 `.to('cuda')`，在没有 Nvidia 显卡的服务器上会抛出 `RuntimeError: CUDA error: no CUDA-capable device is detected`，导致整个服务不可用。

---




