import torch
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusion_model import LDM, VAE, UNetDiffusion


def predict_next_frame(model, input_frame, params):
    """预测下一帧图像"""
    model.eval()
    with torch.no_grad():
        # 预测
        output = model.predict_next_frame(input_frame, params)
    return output


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    vae = VAE(in_channels=3, out_channels=3, z_dim=args.latent_dim)

    denoise_network = UNetDiffusion(
        in_channels=args.latent_dim,
        out_channels=args.latent_dim,
        time_dim=args.time_dim,
        cond_dim=3,
    )

    model = LDM(
        img_channels=3,
        z_channels=args.latent_dim,
        time_dim=args.time_dim,
        cond_dim=3,
        denoise_network=denoise_network,
        vae=vae,
    )

    # 加载预训练模型
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 图像预处理
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # 加载输入图像
    input_img = Image.open(args.input_image).convert("RGB")
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    # 设置参数 (风速, 位置, 角度)
    params_tensor = torch.tensor(
        [
            [
                args.wind_speed / 15.0,  # 归一化风速 (5-15 m/s)
                args.position / 4.0,  # 归一化位置 (0-4)
                args.wind_degree / 315.0,  # 归一化角度 (0-315度)
            ]
        ],
        dtype=torch.float32,
    ).to(device)

    # 预测下一帧
    output_tensor = predict_next_frame(model, input_tensor, params_tensor)

    # 后处理
    output_img = output_tensor[0].cpu().detach()
    output_img = (output_img + 1) / 2.0  # 从[-1,1]转换到[0,1]
    output_img = output_img.permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 显示输入图像
    input_np = input_tensor[0].cpu().detach()
    input_np = (input_np + 1) / 2.0
    input_np = input_np.permute(1, 2, 0).numpy()
    axes[0].imshow(input_np)
    axes[0].set_title("输入帧")
    axes[0].axis("off")

    # 显示预测图像
    axes[1].imshow(output_img)
    axes[1].set_title("预测的下一帧")
    axes[1].axis("off")

    plt.tight_layout()

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, "prediction.png"))

    # 保存预测图像
    output_pil = Image.fromarray((output_img * 255).astype(np.uint8))
    output_pil.save(os.path.join(args.output_dir, "predicted_frame.png"))

    print(f"预测结果已保存到 {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用扩散模型预测污染物扩散下一帧")

    parser.add_argument("--input_image", type=str, required=True, help="输入图像路径")
    parser.add_argument(
        "--output_dir", type=str, default="./predictions", help="输出目录"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/ldm/ldm_final.pt",
        help="模型路径",
    )
    parser.add_argument("--img_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--latent_dim", type=int, default=4, help="VAE潜在空间维度")
    parser.add_argument("--time_dim", type=int, default=256, help="时间嵌入维度")

    # 条件参数
    parser.add_argument(
        "--wind_speed", type=float, default=10.0, help="风速 (5-15 m/s)"
    )
    parser.add_argument("--position", type=int, default=0, help="污染源位置 (0-4)")
    parser.add_argument("--wind_degree", type=int, default=0, help="风向角度 (0-315度)")

    args = parser.parse_args()
    main(args)
