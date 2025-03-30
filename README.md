# PDPM - 基于深度学习的污染物扩散预测模型

## 项目概述

PDPM（Pollutant Diffusion Prediction Model）是一个基于深度学习的污染物扩散预测模型，用于预测给定条件下污染物的扩散过程。本项目采用潜在扩散模型（Latent Diffusion Model, LDM）架构，将变分自编码器（VAE）与去噪扩散概率模型相结合，实现高精度的污染物扩散轨迹预测。

本模型可根据输入的单帧污染物扩散状态图像，结合风速、污染源位置和风向角度等条件参数，预测下一时间步的污染物扩散状态，为环境监测和污染物扩散模拟提供数据驱动的解决方案。

## 研究背景与动机

传统的污染物扩散预测主要依靠复杂的物理模型，计算成本高且对初始条件敏感。随着深度学习技术的发展，基于数据的方法可以通过学习历史扩散数据中的模式来实现高效准确的预测。本项目旨在探索深度学习方法在污染物扩散预测中的应用，通过建立端到端的神经网络模型，实现对污染物扩散过程的快速精确预测。

与传统方法相比，基于深度学习的方法具有以下优势：
- 计算效率高：模型训练完成后，预测过程速度快，适合实时应用
- 泛化能力强：通过学习大量数据，模型可以适应各种不同的条件参数
- 无需详细物理建模：模型可以从数据中自动学习复杂的扩散规律

## 数据集描述

本项目使用的数据集 [基于数据驱动的污染物扩散深度学习模型png版本_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/198102) 
是在特定条件下生成的污染物扩散模拟序列图像：

- **模拟区域**：3km×3km的固定区域
- **污染源位置**：以区域中心为坐标原点，四个象限的中心点和坐标原点构成5个可能的污染源位置
- **风向参数**：8个风向（0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°），其中北风为0°，顺时针旋转
- **风速参数**：3种风速（5m/s, 10m/s, 15m/s）
- **时间序列**：每个模拟案例包含900秒的污染物扩散结果，由181帧组成

文件命名规则：`U*_Pos_*_Deg_***`，其中`U`表示风速，`Pos`表示污染源位置，`Deg`表示风向角度。

## 技术原理

### 扩散模型

扩散模型(Diffusion Models)是一类基于非平衡热力学启发的生成模型，其核心思想是通过模拟物理扩散过程，逐步将数据转化为噪声，然后学习逆向过程从噪声中恢复原始数据。这类模型包含三种主要形式：去噪扩散概率模型(DDPMs)、基于分数的生成模型以及基于随机微分方程的模型。其中，DDPMs因其理论完备性和实践有效性成为当前研究的热点，也是本文研究的重点。

扩散模型的工作机制分为两个互补的阶段：**正向扩散过程**和**反向扩散过程**。正向过程通过逐步添加噪声破坏数据结构，将原始数据转化为高斯噪声；反向过程则通过学习逆转这一噪声化过程，从随机噪声中生成高质量数据样本。这种独特的"扩散-逆扩散"机制使扩散模型在图像生成、数据增强等领域展现出卓越性能，逐渐成为生成式人工智能(AIGC)的核心技术之一。

#### 正向扩散过程

DDPMs的正向过程被建模为一个马尔可夫链，通过T个时间步逐步向数据添加高斯噪声。设z₀为原始数据，在时间步t的转换遵循：

$$ q(z_t|z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta_t}z_{t-1}, \beta_t\mathbf{I}) $$

其中 $\beta_t$ 为噪声调度参数，控制各时间步的噪声添加量。通过递推可得任意时刻t关于初始状态z₀的解析式：

$$z_t = \sqrt{\bar{\beta}_t}z_0 + \sqrt{1-\bar{\beta}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0,\mathbf{I})$$

这里 $ \bar{\beta}_t = \prod_{s=1}^t \beta_s $ 表示累积噪声比例。随着t增大，数据逐渐失去原有结构，最终 $z_T$ 近似为标准高斯分布。

#### 反向扩散过程

反向过程旨在从噪声中重建数据，同样建模为马尔可夫链，但通过学习参数化的转移核实现：

$$ 
p_\theta(z_{t-1}|z_t) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t,t), \Sigma_\theta(z_t,t)) 
$$

其中θ为可学习参数，通常由神经网络实现。反向过程从先验分布 $p(z_T)=\mathcal{N}(0,\mathbf{I})$ 开始，通过逐步去噪最终生成数据样本z₀。

### 训练目标

DDPMs的训练目标是使反向链的联合分布 $p_\theta(z_{0:T})$ 尽可能匹配正向链的时间反转。这通过最小化以下损失函数实现：

$$\mathbb{E}_{t,z_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(z_t,t)\|^2\right]$$

其中 $\epsilon_\theta$ 为噪声预测网络，通常采用U-Net架构。条件扩散模型则扩展为 $\epsilon_\theta(z_t,t,C)$ ，C表示条件变量。

### 扩散模型的优势与应用

相比GAN等传统生成模型，扩散模型具有多项优势：
- **训练稳定性**：避免模式崩溃问题，保证输出多样性
- **生成质量**：通过精细的逐步去噪过程，产生高保真样本
- **理论保障**：基于严格的数学推导，有明确的优化目标
- **灵活扩展**：可结合条件信息，适应多种生成任务

在科学计算领域，扩散模型已成功应用于流体模拟、分子生成等复杂系统的建模。本文研究的有毒气体扩散预测正可借鉴这些思想，通过扩散模型捕捉气体传播的时空动态特性。

### 潜在扩散模型架构

传统扩散模型在高维像素空间直接操作，面临训练收敛慢、计算成本高的问题。稳定扩散(Stable Diffusion)通过**潜在空间扩散**解决了这一挑战，其核心创新包括：

1. 使用变分自编码器(VAE)将图像编码到低维潜在空间，扩散过程在该空间进行
2. 采用U-Net作为噪声预测器，聚合不同层次的多分辨率特征
3. 通过将各层特征图统一缩放到潜在空间的1/4分辨率，实现高效特征融合

这种方法显著降低了计算复杂度，使模型能在消费级硬件上运行，同时保持了生成质量。
本项目采用潜在扩散模型（Latent Diffusion Model）作为核心架构，该架构由两个主要部分组成：变分自编码器（VAE）和去噪扩散网络。

1. **变分自编码器（VAE）**：
   - 负责将高维图像压缩到低维潜在空间，减少计算复杂度
   - 由编码器和解码器组成，学习图像的紧凑表示
   - 在潜在空间中进行扩散过程，提高模型效率

2. **UNet去噪网络**：
   - 采用基于UNet的架构进行噪声预测
   - 包含时间嵌入和条件嵌入，使模型能够处理时序信息并接受外部条件参数
   - 使用跳跃连接和自注意力机制增强特征提取能力

3. **条件控制**：
   - 模型接受风速、污染源位置和风向角度作为条件输入
   - 条件参数通过专门的嵌入层转换为网络可用的表示，并与时间嵌入融合

### 模型训练流程

模型训练分为两个阶段：

1. **VAE预训练**：
   - 首先训练变分自编码器，使其能够有效编码和解码污染物扩散图像
   - 优化目标包括重建损失和KL散度

2. **LDM训练**：
   - 冻结预训练好的VAE参数
   - 训练去噪网络，使其能够在潜在空间中预测下一帧
   - 优化目标包括潜在空间损失和图像空间损失

### 预测机制

预测过程采用单步预测方法：
1. 将输入帧通过VAE编码到潜在空间
2. 使用去噪网络在潜在空间预测下一帧表示
3. 将预测的潜在表示通过VAE解码器还原为图像

## 项目结构

```
PDPM/
├── README.md                  # 项目说明文档
├── main.py                    # 主程序入口
├── dataset.py                 # 数据集加载与处理
├── diffusion_model.py         # 扩散模型定义
├── train.py                   # 模型训练相关函数
├── predict.py                 # 模型预测与推理
├── utils.py                   # 工具函数
└── data1/                     # 数据目录
    └── 基于数据驱动的污染物扩散深度学习模型.txt  # 数据集说明文档
```

## 核心模块详解

### 数据处理模块

数据处理模块在`dataset.py`中实现，主要包括：

- **PollutionDiffusionDataset类**：
  - 加载污染物扩散序列图像
  - 解析场景参数（风速、位置和风向角度）
  - 将图像转换为模型输入格式
  - 支持序列采样，用于时序预测

- **数据加载器函数**：
  - 划分训练集、验证集和测试集
  - 创建DataLoader对象，支持批量处理和并行加载

### 模型定义模块

模型定义在`diffusion_model.py`中实现，包含以下核心组件：

- **变分自编码器（VAE）**：
  - Encoder：将图像映射到潜在空间分布（均值和方差）
  - Decoder：将潜在空间表示重建为图像

- **UNet去噪网络**：
  - 下采样路径：提取多尺度特征
  - 上采样路径：通过跳跃连接恢复空间细节
  - 自注意力层：捕获长距离依赖关系
  - 条件嵌入：处理外部条件参数

- **潜在扩散模型（LDM）**：
  - 整合VAE和去噪网络
  - 实现时间步采样和预测逻辑
  - 提供模型训练和推理接口

### 训练模块

训练逻辑在`train.py`中实现，主要包括：

- **VAE预训练函数**：
  - 优化VAE重建损失和KL散度
  - 定期保存模型检查点
  - 可视化重建效果

- **LDM训练函数**：
  - 冻结VAE参数
  - 实现潜在空间扩散过程
  - 优化潜在空间和图像空间的损失函数
  - 定期评估预测效果

- **可视化函数**：
  - 展示模型预测结果与真实数据对比
  - 评估模型性能

### 预测模块

预测逻辑在`predict.py`中实现，提供了用于模型推理的功能：

- 加载预训练模型
- 接收输入图像和条件参数
- 生成下一帧预测
- 可视化和保存预测结果

## 模型训练与评估

### 训练参数设置

模型训练支持以下关键参数：

- **数据参数**：
  - `--data_dir`：数据目录
  - `--batch_size`：批次大小
  - `--img_size`：图像尺寸

- **模型参数**：
  - `--latent_dim`：VAE潜在空间维度
  - `--time_dim`：时间嵌入维度

- **训练参数**：
  - `--train_vae`：是否训练VAE
  - `--vae_epochs`：VAE训练轮数
  - `--vae_lr`：VAE学习率
  - `--train_ldm`：是否训练LDM
  - `--ldm_epochs`：LDM训练轮数
  - `--ldm_lr`：LDM学习率

- **预训练模型加载**：
  - `--load_vae`：是否加载预训练VAE
  - `--vae_path`：预训练VAE路径
  - `--load_ldm`：是否加载预训练LDM
  - `--ldm_path`：预训练LDM路径

### 训练流程

1. **环境准备**：
   ```bash
   pip install -r requirements.txt
   ```

2. **数据准备**：
   - 将数据集放置在`./data1`目录下
   - 确保数据集目录结构符合要求

3. **VAE预训练**：
   ```bash
   python main.py --train_vae --vae_epochs 20 --vae_lr 1e-4
   ```

4. **LDM训练**：
   ```bash
   python main.py --load_vae --vae_path ./checkpoints/vae/vae_final.pt --train_ldm --ldm_epochs 50 --ldm_lr 1e-5
   ```

5. **模型评估**：
   ```bash
   python main.py --load_vae --load_ldm
   ```

### 模型预测

使用训练好的模型进行预测：

```bash
python predict.py --input_image ./data1/3km/U5_Pos_0_Deg_0/frame_1.png --wind_speed 5 --position 0 --wind_degree 0
```

### 评估指标
模型评估使用以下指标：

#### 图像质量评价指标
峰值信噪比 (PSNR)
- 基于MSE的对数度量，反映图像的重建质量
- 值越高表示预测质量越好
- 通常以dB为单位

#### 污染物扩散特定指标
1. 污染物质量守恒误差
   - 计算预测图像与真实图像中污染物总量的相对误差（假设污染物浓度由图像亮度表示）
   - 反映模型是否保持污染物质量守恒
2. 浓度阈值预测准确率
   - 以特定浓度阈值为界，计算预测的高/低浓度区域与真实区域的重叠程度
   - 类似于分割任务的IoU (Intersection over Union)

## 技术创新点

1. **潜在扩散模型的应用**：
   - 将潜在扩散模型应用于污染物扩散预测，减少计算复杂度
   - 利用VAE压缩表示，提高模型效率

2. **条件控制机制**：
   - 融合风速、污染源位置和风向角度等条件参数
   - 允许模型适应不同环境条件的扩散预测

3. **时空特征学习**：
   - 结合时间嵌入和空间特征提取
   - 捕捉污染物扩散的时空动态特性

## 应用场景

该模型可应用于多种环境监测和污染控制场景：

- **应急响应规划**：快速预测污染物扩散路径，辅助制定应急预案
- **环境影响评估**：模拟不同条件下的污染物扩散情况，评估环境影响
- **城市规划**：分析城市布局对污染物扩散的影响，优化城市设计
- **工业排放管控**：预测工业排放物的扩散轨迹，辅助排放管理

## 未来工作

1. **多尺度模型**：
   - 整合不同空间尺度的扩散模型，提高预测精度
   - 实现从局部到区域尺度的无缝预测

2. **时序预测增强**：
   - 扩展模型以支持长序列预测
   - 引入记忆机制处理长时间依赖

3. **多模态融合**：
   - 整合气象数据、地形数据等多源信息
   - 提高模型在复杂环境下的预测能力

4. **不确定性量化**：
   - 引入概率预测机制
   - 评估预测结果的不确定性范围

## 参考文献

1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

2. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems.

3. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

4. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention.

## 贡献者

- [Rvosuke]
