import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, reduce


# -------------------- 自注意力机制 --------------------
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.1))

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# -------------------- 时间嵌入 --------------------
def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    exponent = (
        -math.log(10000)
        * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        / half_dim
    )
    exponent = exponent.to(device=timesteps.device)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if dim % 2 == 1:  # 如果dim是奇数，填充最后一个值
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    return emb


# -------------------- 残差模块 --------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, use_attention=False):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        if use_attention:
            self.attn = SelfAttention(out_channels)
        else:
            self.attn = nn.Identity()

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb

        h = self.block2(h)

        # 应用注意力（如果使用）
        h = rearrange(h, "b c h w -> b (h w) c")
        h = self.attn(h)
        h = rearrange(h, "b (h w) c -> b c h w", h=int(math.sqrt(h.shape[1])))

        return h + self.shortcut(x)


# -------------------- 条件嵌入 --------------------
class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, emb_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, cond):
        return self.linear(cond)


# -------------------- UNet噪声预测器 --------------------
class UNetDiffusion(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_dim=256,
        cond_dim=3,
        channels=[64, 128, 256, 512],
        attention_layers=[False, True, True, False],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_dim = time_dim

        # 时间嵌入层
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # 条件嵌入层
        self.cond_mlp = ConditionEmbedding(cond_dim, time_dim)

        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder (下采样)
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            channels[i], channels[i], time_dim, attention_layers[i]
                        ),
                        ResidualBlock(
                            channels[i], channels[i + 1], time_dim, attention_layers[i]
                        ),
                        nn.MaxPool2d(2),
                    ]
                )
            )

        # 中间层
        mid_channels = channels[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, time_dim, True)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, time_dim, False)

        # Decoder (上采样)
        self.ups = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i in range(len(channels) - 1):
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            reversed_channels[i] * 2,
                            reversed_channels[i],
                            time_dim,
                            attention_layers[-(i + 1)],
                        ),
                        ResidualBlock(
                            reversed_channels[i],
                            reversed_channels[i + 1],
                            time_dim,
                            attention_layers[-(i + 1)],
                        ),
                        nn.ConvTranspose2d(
                            reversed_channels[i + 1], reversed_channels[i + 1], 4, 2, 1
                        ),
                    ]
                )
            )

        # 输出层
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1),
        )

    def forward(self, x, time, cond):
        # 时间嵌入
        t = sinusoidal_embedding(time, self.time_dim)
        t = self.time_mlp(t)

        # 条件嵌入并与时间嵌入融合
        c = self.cond_mlp(cond)
        t = t + c

        # 初始特征
        x = self.init_conv(x)

        # 保存每一层的特征用于跳跃连接
        skips = []

        # 下采样路径
        for i, down_block in enumerate(self.downs):
            res1, res2, downsample = down_block
            x = res1(x, t)
            x = res2(x, t)
            skips.append(x)
            if i < len(self.downs) - 1:
                # 仅在最后一层之前下采样
                x = downsample(x)

        # 中间层
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # 上采样路径
        for i, up_block in enumerate(self.ups):
            res1, res2, upsample = up_block
            skip = skips.pop()

            x = torch.cat([x, skip], dim=1)
            x = res1(x, t)
            x = res2(x, t)
            if i < len(self.ups) - 1:
                # 仅在最后一层之前上采样
                x = upsample(x)

        # 输出层
        x = self.output_conv(x)
        return x


# -------------------- VAE编码器/解码器 --------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=3, z_dim=4, channels=[32, 64, 128, 256]):
        super().__init__()

        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder模块
        self.encoder_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1),
                    nn.GroupNorm(
                        32 if channels[i + 1] >= 32 else channels[i + 1],
                        channels[i + 1],
                    ),
                    nn.SiLU(),
                )
            )

        # 最终输出（均值和方差）
        self.out = nn.Conv2d(channels[-1], 2 * z_dim, 1)

    def forward(self, x):
        x = self.init_conv(x)

        # 编码
        for layer in self.encoder_layers:
            x = layer(x)

        # 输出均值和方差
        out = self.out(x)
        mu, log_var = torch.chunk(out, 2, dim=1)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, out_channels=3, z_dim=4, channels=[256, 128, 64, 32]):
        super().__init__()

        # 初始卷积
        self.init_conv = nn.Conv2d(z_dim, channels[0], 3, padding=1)

        # Decoder模块
        self.decoder_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        channels[i], channels[i + 1], 4, stride=2, padding=1
                    ),
                    nn.GroupNorm(
                        32 if channels[i + 1] >= 32 else channels[i + 1],
                        channels[i + 1],
                    ),
                    nn.SiLU(),
                )
            )

        # 最终输出
        self.out = nn.Sequential(
            nn.Conv2d(channels[-1], out_channels, 3, padding=1), nn.Tanh()
        )

    def forward(self, x):
        x = self.init_conv(x)

        # 解码
        for layer in self.decoder_layers:
            x = layer(x)

        x = self.out(x)
        return x


class VAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, z_dim=4):
        super().__init__()
        self.encoder = Encoder(in_channels, z_dim)
        self.decoder = Decoder(out_channels, z_dim)

    def forward(self, x):
        # 编码
        mu, log_var = self.encoder(x)

        # 重参数化
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # 解码
        x_recon = self.decoder(z)

        return x_recon, mu, log_var

    def encode(self, x):
        mu, log_var = self.encoder(x)
        # 重参数化
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)


# -------------------- 扩散模型 --------------------
class LDM(nn.Module):
    def __init__(
        self,
        img_channels=3,
        z_channels=4,
        time_dim=256,
        cond_dim=3,
        denoise_network=None,
        vae=None,
    ):
        super().__init__()

        # 设置参数
        self.img_channels = img_channels
        self.z_channels = z_channels

        # VAE模型 (用于潜在空间编解码)
        self.vae = vae if vae else VAE(img_channels, img_channels, z_channels)

        # 噪声预测模型
        if denoise_network is None:
            self.denoise_network = UNetDiffusion(
                in_channels=z_channels,
                out_channels=z_channels,
                time_dim=time_dim,
                cond_dim=cond_dim,
            )
        else:
            self.denoise_network = denoise_network

    def encode_to_latent(self, x):
        """将图像编码到潜在空间"""
        with torch.no_grad():
            return self.vae.encode(x)

    def decode_from_latent(self, z):
        """从潜在空间解码到图像"""
        with torch.no_grad():
            return self.vae.decode(z)

    def forward(self, x_t0, cond, t=None):
        """
        单步扩散预测 - 从当前帧(t0)预测下一帧(t1)

        Args:
            x_t0 (torch.Tensor): 当前帧图像
            cond (torch.Tensor): 条件参数 (风速、位置、角度)
            t (torch.Tensor, optional): 时间步, 如果不提供则随机生成

        Returns:
            torch.Tensor: 预测的下一帧图像
        """
        batch_size = x_t0.shape[0]
        device = x_t0.device

        # 如果未提供时间步，则随机生成(或固定为单一时间步)
        if t is None:
            # 使用固定时间步更简单更稳定
            t = torch.ones(batch_size, device=device, dtype=torch.long)

        # 将输入图像编码到潜在空间
        z_t0 = self.encode_to_latent(x_t0)

        # 使用去噪模型预测下一帧的潜在表示
        z_t1_pred = self.denoise_network(z_t0, t, cond)

        # 将预测的潜在表示解码回图像空间
        x_t1_pred = self.decode_from_latent(z_t1_pred)

        return x_t1_pred, z_t0, z_t1_pred

    def predict_next_frame(self, x_t0, cond):
        """从当前帧预测下一帧 (用于推理)"""
        device = x_t0.device
        batch_size = x_t0.shape[0]

        # 固定时间步为1 (单步预测)
        t = torch.ones(batch_size, device=device, dtype=torch.long)

        # 编码到潜在空间
        with torch.no_grad():
            z_t0 = self.encode_to_latent(x_t0)

            # 预测下一帧的潜在表示
            z_t1_pred = self.denoise_network(z_t0, t, cond)

            # 解码回图像空间
            x_t1_pred = self.decode_from_latent(z_t1_pred)

        return x_t1_pred

    def get_vae_loss(self, x):
        """计算VAE重建损失用于预训练VAE"""
        # 前向传播获取重建结果和分布参数
        x_recon, mu, log_var = self.vae(x)

        # 重建损失 (MSE)
        recon_loss = F.mse_loss(x_recon, x)

        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / (x.shape[0] * x.shape[2] * x.shape[3])  # 归一化

        # 总损失 (可以调整权重)
        total_loss = recon_loss + 0.001 * kl_loss
        return total_loss, recon_loss, kl_loss
