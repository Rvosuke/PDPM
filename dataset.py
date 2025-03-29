import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import re


class PollutionDiffusionDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=2, img_size=256):
        """
        污染物扩散数据集加载器 - 适用于扩散模型

        Args:
            root_dir (str): 数据根目录，包含所有场景文件夹
            transform (callable, optional): 可选的图像转换函数
            sequence_length (int): 每个样本包含的帧数，默认为2（输入1帧，预测1帧）
            img_size (int): 图像大小
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.img_size = img_size

        # 从tree.txt读取场景列表
        with open(os.path.join(root_dir, "tree.txt"), "r") as f:
            self.scenarios = [line.strip() for line in f if line.strip()]

        # 构建样本列表
        self.samples = []
        self.timestep = []
        for scenario in self.scenarios:
            scenario_dir = os.path.join(root_dir, "3km", scenario)
            if not os.path.isdir(scenario_dir):
                continue

            # 获取该场景下的所有帧
            frames = []
            for i in range(1, 182):  # 从文档知道每个场景有181帧
                frame_path = os.path.join(scenario_dir, f"frame_{i}.png")
                if os.path.exists(frame_path):
                    frames.append(frame_path)

            # 为每个可能的输入/输出序列创建样本
            for i in range(len(frames) - sequence_length + 1):
                self.samples.append((frames[i : i + sequence_length], scenario))
                self.timestep.append(i)

        # 解析场景参数（风速、位置和角度）用于条件输入
        self.scenario_params = {}
        for scenario in self.scenarios:
            match = re.search(r"U(\d+)_Pos_(\d+)_Deg_(\d+)", scenario)
            if match:
                wind_speed = int(match.group(1))
                position = int(match.group(2))
                degree = int(match.group(3))
                self.scenario_params[scenario] = {
                    "wind_speed": wind_speed,
                    "position": position,
                    "degree": degree,
                }

        # 默认转换
        self.default_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取一个序列样本
        sequence, scenario_name = self.samples[idx]
        timestep = torch.tensor(
            self.timestep[idx],
            dtype=torch.float32,
        )

        # 加载图像
        frames = []
        for frame_path in sequence:
            img = Image.open(frame_path).convert("RGB")  # 保持彩色图像
            if self.transform:
                img = self.transform(img)
            else:
                # 默认转换
                img = self.default_transform(img)
            frames.append(img)

        # 提取场景参数
        params = self.scenario_params.get(
            scenario_name, {"wind_speed": 0, "position": 0, "degree": 0}
        )

        # 标准化参数
        normalized_params = torch.tensor(
            [
                params["wind_speed"] / 15.0,  # 风速范围 5-15
                params["position"] / 4.0,  # 位置范围 0-4
                params["degree"] / 315.0,  # 角度范围 0-315
            ],
            dtype=torch.float32,
        )

        # 对于扩散模型，返回输入帧和目标帧
        input_frame = frames[0]
        target_frame = frames[-1]

        return input_frame, target_frame, normalized_params, timestep


def get_data_loaders(
    data_dir,
    batch_size=16,
    train_ratio=0.8,
    val_ratio=0.1,
    sequence_length=2,
    num_workers=4,
    img_size=256,
):
    """
    创建训练、验证和测试数据加载器

    Args:
        data_dir (str): 数据根目录
        batch_size (int): 批次大小
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        sequence_length (int): 序列长度
        num_workers (int): 数据加载进程数
        img_size (int): 图像大小

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 创建数据集
    dataset = PollutionDiffusionDataset(
        root_dir=data_dir, sequence_length=sequence_length, img_size=img_size
    )

    # 计算数据集划分
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # 随机划分数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
