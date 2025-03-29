import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import matplotlib.pyplot as plt

from diffusion_model import LDM, VAE, UNetDiffusion
from dataset import get_data_loaders
from train import train_vae, train_ldm, visualize_predictions, evaluate_model
from utils import count_parameters


def main(args):
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,  # 输入1帧，输出1帧
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    print(f"数据加载器创建完成，训练集样本数: {len(train_loader.dataset)}")

    # 创建模型
    vae = VAE(
        in_channels=3,  # RGB图像
        out_channels=3,
        z_dim=args.latent_dim,
    )

    denoise_network = UNetDiffusion(
        in_channels=args.latent_dim,  # 潜在空间维度
        out_channels=args.latent_dim,
        time_dim=args.time_dim,  # 时间嵌入维度
        cond_dim=3,  # 风速、位置、角度
        channels=[64, 128, 256, 512],
        attention_layers=[False, True, True, False],
    )

    # 创建LDM模型
    model = LDM(
        img_channels=3,  # RGB图像
        z_channels=args.latent_dim,
        time_dim=args.time_dim,
        cond_dim=3,  # 风速、位置、角度
        denoise_network=denoise_network,
        vae=vae,
    )

    print(f"模型创建完成，参数量: {count_parameters(model) / 1e6:.2f}M")

    # 检查是否需要加载预训练的VAE
    if args.load_vae:
        vae_path = args.vae_path
        if os.path.exists(vae_path):
            model.vae.load_state_dict(
                torch.load(vae_path, map_location=device, weights_only=True)
            )
            print(f"加载预训练VAE模型: {vae_path}")
        else:
            print(f"找不到预训练VAE模型: {vae_path}，将从头训练")

    # 检查是否需要加载预训练的LDM
    if args.load_ldm:
        ldm_path = args.ldm_path
        if os.path.exists(ldm_path):
            checkpoint = torch.load(ldm_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"加载预训练LDM模型: {ldm_path}")
        else:
            print(f"找不到预训练LDM模型: {ldm_path}，将从头训练")

    # 训练VAE
    if args.train_vae and not args.load_vae:
        print("开始训练VAE...")
        # VAE优化器
        vae_optimizer = optim.Adam(model.vae.parameters(), lr=args.vae_lr)

        # 训练VAE
        train_vae(
            model, train_loader, vae_optimizer, device, num_epochs=args.vae_epochs
        )

        # 保存训练后的VAE
        os.makedirs("./checkpoints/vae", exist_ok=True)
        torch.save(model.vae.state_dict(), "./checkpoints/vae/vae_final.pt")
        print("VAE训练完成并保存")

    # 冻结VAE参数
    for param in model.vae.parameters():
        param.requires_grad = False

    # 训练LDM
    if args.train_ldm:
        print("开始训练LDM...")
        # LDM优化器 - 只优化去噪网络
        ldm_optimizer = optim.AdamW(
            model.denoise_network.parameters(),
            lr=args.ldm_lr,
            weight_decay=args.weight_decay,
        )

        # 训练LDM
        loss_history, metrics_history = train_ldm(
            model,
            train_loader,
            val_loader,
            ldm_optimizer,
            device,
            num_epochs=args.ldm_epochs,
        )
        # 可视化训练历史

        # 绘制loss历史
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.title("Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        os.makedirs("./results", exist_ok=True)
        plt.savefig("./results/loss_history.png")
        plt.close()

        # 绘制metrics历史
        plt.figure(figsize=(12, 6))
        for metric_name, values in metrics_history.items():
            plt.plot(values, label=metric_name)
        plt.title("Validation Metrics History")
        plt.xlabel("Epoch/5")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig("./results/metrics_history.png")
        plt.close()
        # 保存训练后的LDM
        os.makedirs("./checkpoints/ldm", exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": ldm_optimizer.state_dict(),
            },
            "./checkpoints/ldm/ldm_final.pt",
        )
        print("LDM训练完成并保存")

    # 测试模型
    print("开始评估模型...")
    model.eval()
    metrics = evaluate_model(model, test_loader, device)
    print(f"测试指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    # Save metrics to a file
    with open("./results/test_metrics.txt", "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.6f}\n")
    visualize_predictions(model, test_loader, "test", device, num_samples=8)
    print("模型评估完成，结果已保存到'results'文件夹")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="污染物扩散预测扩散模型")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data1", help="数据目录")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--sequence_length", type=int, default=2, help="序列长度")
    parser.add_argument("--img_size", type=int, default=256, help="图像尺寸")

    # 模型参数
    parser.add_argument("--latent_dim", type=int, default=4, help="VAE潜在空间维度")
    parser.add_argument("--time_dim", type=int, default=256, help="时间嵌入维度")

    # 训练参数
    parser.add_argument("--train_vae", action="store_true", help="是否训练VAE")
    parser.add_argument("--vae_epochs", type=int, default=20, help="VAE训练轮数")
    parser.add_argument("--vae_lr", type=float, default=1e-4, help="VAE学习率")

    parser.add_argument("--train_ldm", action="store_true", help="是否训练LDM")
    parser.add_argument("--ldm_epochs", type=int, default=50, help="LDM训练轮数")
    parser.add_argument("--ldm_lr", type=float, default=1e-5, help="LDM学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")

    # 加载预训练模型
    parser.add_argument("--load_vae", action="store_true", help="是否加载预训练VAE")
    parser.add_argument(
        "--vae_path",
        type=str,
        default="./checkpoints/vae/vae_final.pt",
        help="预训练VAE路径",
    )
    parser.add_argument("--load_ldm", action="store_true", help="是否加载预训练LDM")
    parser.add_argument(
        "--ldm_path",
        type=str,
        default="./checkpoints/ldm/ldm_final.pt",
        help="预训练LDM路径",
    )

    args = parser.parse_args()
    main(args)
