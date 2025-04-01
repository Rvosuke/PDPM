from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils import evaluate_prediction


def train_vae(model, train_dataloader, optimizer, device, num_epochs=10):
    """
    预训练VAE模型

    Args:
        model: LDM模型 (使用内部VAE组件)
        train_dataloader: 训练数据加载器
        optimizer: 优化器
        device: 训练设备 (cuda/cpu)
        num_epochs: 训练轮数
    """
    model.to(device)
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"VAE预训练 Epoch {epoch+1}/{num_epochs}",
        )

        for batch_id, (inputs, targets, _, _) in pbar:
            # 将数据移到指定设备
            inputs = inputs.to(device)

            # 清除梯度
            optimizer.zero_grad()

            # 计算VAE损失
            total_loss, recon_loss, kl_loss = model.get_vae_loss(inputs)

            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 更新统计信息
            current_loss = total_loss.item()
            epoch_loss += current_loss

            # 更新进度条
            pbar.set_postfix(
                {
                    "total_loss": f"{current_loss:.6f}",
                    "recon_loss": f"{recon_loss.item():.6f}",
                    "kl_loss": f"{kl_loss.item():.6f}",
                }
            )

        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        loss_history.append(avg_epoch_loss)

        print(
            f"VAE预训练 Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.6f}"
        )

        # 保存VAE模型
        if (epoch + 1) % 5 == 0:
            save_dir = "./checkpoints/vae"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                model.vae.state_dict(),
                f"{save_dir}/vae_epoch_{epoch+1}.pt",
            )

            # 可视化VAE重建效果
            visualize_vae_reconstruction(model, train_dataloader, epoch, device)

    return loss_history


def train_ldm(
    model, train_dataloader, val_dataloader, optimizer, device, num_epochs=50
):
    """
    训练Latent Diffusion Model

    Args:
        model: LDM模型
        train_dataloader: 训练数据加载器
        optimizer: 优化器
        device: 训练设备 (cuda/cpu)
        num_epochs: 训练轮数
    """
    model.to(device)
    loss_history = []
    metrics_history = []
    best_psnr = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"LDM训练 Epoch {epoch+1}/{num_epochs}",
        )

        for batch_id, (inputs, targets, params, t) in pbar:
            # 将数据移到指定设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            params = params.to(device)
            t = t.to(device)

            batch_size = inputs.shape[0]

            # 随机生成时间步 (介于1到1000之间)
            # t = torch.randint(1, 1000, (batch_size,), device=device).long()

            # 清除梯度
            optimizer.zero_grad()

            # 前向传播
            pred_next_frame, latent_input, latent_pred = model(inputs, params, t)

            # 计算潜在空间损失和图像空间损失
            latent_target = model.encode_to_latent(targets)
            latent_loss = F.mse_loss(latent_pred, latent_target)

            image_loss = F.mse_loss(pred_next_frame, targets)

            # 总损失 (可以调整权重)
            total_loss = image_loss + 0.1 * latent_loss

            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 更新统计信息
            current_loss = total_loss.item()
            epoch_loss += current_loss

            # 更新进度条
            pbar.set_postfix(
                {
                    "total": f"{current_loss:.6f}",
                    "img_loss": f"{image_loss.item():.6f}",
                    "latent_loss": f"{latent_loss.item():.6f}",
                }
            )

        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        loss_history.append(avg_epoch_loss)

        print(
            f"LDM训练 Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.6f}"
        )

        # 每N个epoch评估一次
        if (epoch + 1) % 1 == 0:
            metrics = evaluate_model(model, val_dataloader, device)
            print(f"Epoch {epoch+1} 验证指标:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.6f}")
            metrics_history.append(metrics)
            # 可以使用TensorBoard或其他工具记录指标

            # 保存最佳模型
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                save_dir = "./checkpoints/ldm"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                    },
                    f"{save_dir}/ldm_best.pt",
                )

            # 每5个epoch可视化预测效果
            visualize_predictions(model, val_dataloader, epoch, device)

    return loss_history, metrics_history


def visualize_vae_reconstruction(model, dataloader, epoch, device, num_samples=4):
    """可视化VAE重建效果"""
    model.eval()

    # 获取一些样本
    inputs, _, _, _ = next(iter(dataloader))
    inputs = inputs[:num_samples].to(device)

    with torch.no_grad():
        # 获取重建结果
        recon, _, _ = model.vae(inputs)

    # 将结果转换为numpy数组用于可视化
    inputs = inputs.cpu().numpy()
    recon = recon.cpu().numpy()

    # 规范化到[0,1]以便可视化
    inputs = (inputs + 1) / 2.0
    recon = (recon + 1) / 2.0

    # 创建图表
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))

    for i in range(num_samples):
        # 显示原始图像
        axes[0, i].imshow(np.transpose(inputs[i], (1, 2, 0)))
        axes[0, i].set_title("source")
        axes[0, i].axis("off")

        # 显示重建图像
        axes[1, i].imshow(np.transpose(recon[i], (1, 2, 0)))
        axes[1, i].set_title("VAE Reconstruction")
        axes[1, i].axis("off")

    plt.tight_layout()

    # 保存图像
    os.makedirs("./results", exist_ok=True)
    plt.savefig(f"./results/vae_reconstruction_{epoch}.png", dpi=200)
    plt.close()


def visualize_predictions(model, dataloader, epoch, device, num_samples=4):
    """可视化模型预测结果"""
    model.eval()

    # 获取一些样本
    inputs, targets, params, _ = next(iter(dataloader))
    inputs = inputs[:num_samples].to(device)
    targets = targets[:num_samples].to(device)
    params = params[:num_samples].to(device)

    with torch.no_grad():
        # 获取预测结果
        preds = model.predict_next_frame(inputs, params)

    # 将结果转换为numpy数组用于可视化
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()

    # 规范化到[0,1]以便可视化
    inputs = (inputs + 1) / 2.0
    targets = (targets + 1) / 2.0
    preds = (preds + 1) / 2.0

    # 创建图表
    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9))

    for i in range(num_samples):
        # 显示输入图像
        axes[0, i].imshow(np.transpose(inputs[i], (1, 2, 0)))
        axes[0, i].set_title("Input")
        axes[0, i].axis("off")

        # 显示目标图像
        axes[1, i].imshow(np.transpose(targets[i], (1, 2, 0)))
        axes[1, i].set_title("Target")
        axes[1, i].axis("off")

        # 显示预测图像
        axes[2, i].imshow(np.transpose(preds[i], (1, 2, 0)))
        axes[2, i].set_title("Prediction")
        axes[2, i].axis("off")

    plt.tight_layout()

    # 保存图像
    os.makedirs("./results", exist_ok=True)
    plt.savefig(f"./results/frame_predictions_{epoch}.png", dpi=200)
    plt.close()


def evaluate_model(model, val_loader, device):
    model.eval()
    all_metrics = defaultdict(list)

    with torch.no_grad():
        for inputs, targets, params, _ in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            params = params.to(device)

            # 预测
            predictions = model.predict_next_frame(inputs, params)

            # 计算评估指标
            batch_metrics = evaluate_prediction(predictions, targets)

            # 累积批次评估结果
            for k, v in batch_metrics.items():
                all_metrics[k].append(v)

    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

    return avg_metrics
