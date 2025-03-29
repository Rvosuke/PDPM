import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def count_parameters(model):
    """计算模型的参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_model_output(model, input_tensor, target_tensor=None, figsize=(12, 5)):
    """可视化模型的输出结果

    Args:
        model (nn.Module): PyTorch模型
        input_tensor (torch.Tensor): 输入张量, shape应为[B, C, H, W]
        target_tensor (torch.Tensor, optional): 目标张量用于对比, shape应为[B, C, H, W]
        figsize (tuple): 图像尺寸
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    # 选择第一个样本进行可视化
    sample_input = input_tensor[0].cpu().numpy()
    sample_output = output[0].cpu().numpy()

    plt.figure(figsize=figsize)

    # 可视化输入通道
    for i in range(min(sample_input.shape[0], 3)):
        plt.subplot(2, 3, i + 1)
        plt.title(f"Input Channel {i}")
        plt.imshow(sample_input[i], cmap="viridis")
        plt.colorbar()

    # 可视化输出通道
    for i in range(min(sample_output.shape[0], 3)):
        plt.subplot(2, 3, i + 4)
        plt.title(f"Output Channel {i}")
        plt.imshow(sample_output[i], cmap="viridis")
        plt.colorbar()

    plt.tight_layout()
    plt.show()

    if target_tensor is not None:
        sample_target = target_tensor[0].cpu().numpy()
        plt.figure(figsize=figsize)

        # 计算误差
        error = sample_output - sample_target

        # 可视化目标和误差
        for i in range(min(sample_target.shape[0], 3)):
            plt.subplot(2, 3, i + 1)
            plt.title(f"Target Channel {i}")
            plt.imshow(sample_target[i], cmap="viridis")
            plt.colorbar()

            plt.subplot(2, 3, i + 4)
            plt.title(f"Error Channel {i}")
            plt.imshow(error[i], cmap="RdBu_r")
            plt.colorbar()

        plt.tight_layout()
        plt.show()


def evaluate_prediction(pred_frames, true_frames, threshold=0.5):
    """
    评估预测结果的质量

    Args:
        pred_frames: 预测的帧序列 [batch, time, channels, height, width]
        true_frames: 真实的帧序列 [batch, time, channels, height, width]
        threshold: 用于计算IoU的浓度阈值

    Returns:
        dict: 包含各项评价指标的字典
    """
    metrics = {}

    # MSE
    mse = F.mse_loss(pred_frames, true_frames).item()

    # PSNR
    psnr = 10 * torch.log10(1 / mse).item() if mse > 0 else float("inf")
    metrics["psnr"] = psnr

    # 污染物总量差异（假设污染物浓度由图像亮度表示）
    pred_mass = pred_frames.sum().item()
    true_mass = true_frames.sum().item()
    mass_error = abs(pred_mass - true_mass) / true_mass
    metrics["mass_error"] = mass_error

    # 浓度阈值IoU
    pred_binary = (pred_frames > threshold).float()
    true_binary = (true_frames > threshold).float()
    intersection = (pred_binary * true_binary).sum().item()
    union = (pred_binary + true_binary).clamp(0, 1).sum().item()
    iou = intersection / union if union > 0 else 0
    metrics["iou"] = iou

    return metrics
