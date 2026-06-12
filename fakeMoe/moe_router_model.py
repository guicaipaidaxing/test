import math
import random
import io

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageOps
from torchvision import transforms, models


# =========================================================
# 1. 外部数据预处理
# =========================================================

class ConvertToRGB:
    """
    作用：
        将灰度图、RGBA 图、P 模式图等统一转为 RGB。

    原因：
        后续模型默认输入为 3 通道 RGB 图像。
    """
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


class PadToSquare:
    """
    作用：
        保持原始比例，将图像 padding 成正方形。

    原因：
        直接 Resize 会改变图像比例，可能影响频域、边缘和压缩痕迹。
    """
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size

        if w == h:
            return img

        max_side = max(w, h)

        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top

        img = ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill
        )

        return img


class RandomJPEGCompression:
    """
    作用：
        训练时随机 JPEG 压缩增强。

    原因：
        图像取证任务中，压缩质量会影响噪声、DCT 和频域特征。
        加入随机 JPEG 压缩可以提高模型对不同压缩历史的鲁棒性。

    注意：
        只用于训练，不用于验证和测试。
    """
    def __init__(self, quality_range=(60, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        quality = random.randint(
            self.quality_range[0],
            self.quality_range[1]
        )

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        img = Image.open(buffer).convert("RGB")

        return img


def check_img_size(img_size):
    """
    作用：
        检查图像尺寸是否能被 8 整除。

    原因：
        DCT Expert 使用 8x8 block DCT。
    """
    if img_size % 8 != 0:
        raise ValueError(
            f"img_size={img_size} is invalid. "
            f"img_size must be divisible by 8 for DCT branch."
        )


def build_transforms(img_size):
    """
    作用：
        统一构建 train / val / test transform。

    train_transform:
        用于训练，有轻量增强。

    val_transform / test_transform:
        用于验证和测试，不使用随机增强，保证结果稳定。
    """
    check_img_size(img_size)

    train_transform = transforms.Compose([
        ConvertToRGB(),
        PadToSquare(fill=0),
        transforms.Resize((img_size, img_size)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.08,
                hue=0.01
            )
        ], p=0.3),

        RandomJPEGCompression(quality_range=(60, 100), p=0.5),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.15),

        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        ConvertToRGB(),
        PadToSquare(fill=0),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        ConvertToRGB(),
        PadToSquare(fill=0),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    return train_transform, val_transform, test_transform


# =========================================================
# 2. ResNet18 Backbone
# =========================================================

class ResNet18Feature(nn.Module):
    """
    作用：
        ResNet18 特征提取器。

    支持：
        不同输入通道数。

    输出：
        512 维全局特征。
    """
    def __init__(self, in_channels=3, pretrained=False):
        super().__init__()

        if pretrained and in_channels == 3:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = models.resnet18(weights=weights)

        if in_channels != 3:
            old_conv = model.conv1

            model.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )

        model.fc = nn.Identity()

        self.backbone = model

    def forward(self, x):
        return self.backbone(x)


# =========================================================
# 3. RGB Expert 预处理
# =========================================================

class RGBNormalizePreprocess(nn.Module):
    """
    作用：
        对 RGB Expert 输入做 ImageNet Normalize。

    原因：
        RGB Expert 使用 ImageNet 预训练 ResNet18。
        输入需要匹配预训练模型的归一化方式。
    """
    def __init__(self):
        super().__init__()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


# =========================================================
# 4. TruFor-style Noise Expert
# =========================================================

class BayarConv2d(nn.Module):
    """
    作用：
        可学习约束卷积。

    参考思想：
        Bayar-Stamm constrained convolution
        TruFor / Noiseprint 类 forensic trace learning

    约束目标：
        中心权重为 -1，
        周围权重归一化，使卷积更偏向预测误差 / 噪声残差，
        而不是普通图像内容。

    输入：
        [B, 3, H, W]

    输出：
        [B, out_channels, H, W]
    """
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5):
        super().__init__()

        assert kernel_size % 2 == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.raw_weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            ) * 0.01
        )

        center = kernel_size // 2

        mask = torch.ones(1, 1, kernel_size, kernel_size)
        mask[:, :, center, center] = 0.0

        center_mask = torch.zeros(1, 1, kernel_size, kernel_size)
        center_mask[:, :, center, center] = 1.0

        self.register_buffer("mask", mask)
        self.register_buffer("center_mask", center_mask)

    def forward(self, x):
        w = self.raw_weight * self.mask

        denom = w.sum(dim=(2, 3), keepdim=True)

        # 避免分母过小导致数值不稳定
        denom = torch.where(
            torch.abs(denom) < 1e-6,
            torch.ones_like(denom) * 1e-6,
            denom
        )

        w = w / denom
        w = w - self.center_mask

        out = F.conv2d(
            x,
            w,
            bias=None,
            stride=1,
            padding=self.kernel_size // 2
        )

        return out


class SRMBank(nn.Module):
    """
    作用：
        固定 SRM 高通滤波器组。

    参考思想：
        SRM Rich Model
        Noiseprint
        TruFor forensic trace

    主要用于：
        提取噪声残差、局部不一致、相机指纹异常。
    """
    def __init__(self):
        super().__init__()

        kernels = []

        k1 = torch.tensor([
            [0,  0,  0,  0, 0],
            [0, -1,  2, -1, 0],
            [0,  2, -4,  2, 0],
            [0, -1,  2, -1, 0],
            [0,  0,  0,  0, 0]
        ], dtype=torch.float32)

        k2 = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32)

        k3 = torch.tensor([
            [0,  0,  0,  0, 0],
            [0,  0,  0,  0, 0],
            [0,  1, -2,  1, 0],
            [0,  0,  0,  0, 0],
            [0,  0,  0,  0, 0]
        ], dtype=torch.float32)

        k4 = torch.tensor([
            [0,  0,  0],
            [1, -2,  1],
            [0,  0,  0]
        ], dtype=torch.float32)

        k5 = torch.tensor([
            [0,  1,  0],
            [0, -2,  0],
            [0,  1,  0]
        ], dtype=torch.float32)

        kernels.append(k1)
        kernels.append(k2)
        kernels.append(k3)
        kernels.append(F.pad(k4, (1, 1, 1, 1)))
        kernels.append(F.pad(k5, (1, 1, 1, 1)))

        kernels = torch.stack(kernels, dim=0)
        kernels = kernels.unsqueeze(1)

        self.register_buffer("weight", kernels)

    def forward(self, x):
        gray = (
            0.299 * x[:, 0:1] +
            0.587 * x[:, 1:2] +
            0.114 * x[:, 2:3]
        )

        residual = F.conv2d(gray, self.weight, padding=2)
        residual = torch.clamp(residual, -3.0, 3.0)

        return residual


class TruForNoisePreprocess(nn.Module):
    """
    作用：
        TruFor-style noise residual 输入。

    组成：
        1. 固定 SRM residual，5 个通道
        2. 可学习 Bayar residual，3 个通道

    输出：
        [B, 8, H, W]
    """
    def __init__(self):
        super().__init__()

        self.srm = SRMBank()

        self.bayar = BayarConv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=5
        )

    def normalize_per_channel(self, x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (x - mean) / std

    def forward(self, x):
        srm_res = self.srm(x)
        bayar_res = self.bayar(x)

        srm_res = self.normalize_per_channel(srm_res)
        bayar_res = self.normalize_per_channel(bayar_res)

        out = torch.cat([srm_res, bayar_res], dim=1)

        return out


# =========================================================
# 5. FFT Frequency Expert
# =========================================================

class FFTPreprocess(nn.Module):
    """
    作用：
        提取频域幅度谱和频域高频残差。

    主要用于：
        DMG / AI 生成图检测。
        生成图可能存在频谱异常、周期性 artifact、高频分布异常。

    输入：
        [B, 3, H, W]

    输出：
        [B, 6, H, W]
        其中：
            3 channel log amplitude
            3 channel high-frequency residual
    """
    def normalize_per_channel(self, x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (x - mean) / std

    def forward(self, x):
        fft = torch.fft.fft2(x, norm="ortho")
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        amp = torch.log1p(torch.abs(fft))
        amp = self.normalize_per_channel(amp)

        smooth = F.avg_pool2d(
            amp,
            kernel_size=7,
            stride=1,
            padding=3
        )

        high_freq_res = amp - smooth
        high_freq_res = self.normalize_per_channel(high_freq_res)

        out = torch.cat([amp, high_freq_res], dim=1)

        return out


# =========================================================
# 6. CAT-Net-inspired DCT / JPEG Expert
# =========================================================

def build_zigzag_indices(N=8):
    """
    作用：
        构建 8x8 DCT zigzag 顺序。

    DCT 中：
        左上角为 DC / 低频。
        越靠右下越偏高频。
    """
    indices = []

    for s in range(2 * N - 1):
        if s % 2 == 0:
            for i in range(s, -1, -1):
                j = s - i
                if i < N and j < N:
                    indices.append((i, j))
        else:
            for j in range(s, -1, -1):
                i = s - j
                if i < N and j < N:
                    indices.append((i, j))

    return indices


class CATDCTPreprocess(nn.Module):
    """
    作用：
        CAT-Net-inspired DCT coefficient feature。

    与普通 DCT 分支区别：
        不直接使用全部 64 个 DCT 系数。
        去掉 DC，选择 AC / 中高频系数。

    原因：
        DC / 极低频更多表示图像内容；
        AC / 中高频更容易携带 JPEG 压缩痕迹、块效应、编辑异常。

    输入：
        [B, 3, H, W]

    输出：
        [B, 3 * num_coeff, H/8, W/8]
    """
    def __init__(self, block_size=8, num_coeff=32):
        super().__init__()

        self.block_size = block_size
        self.num_coeff = num_coeff

        dct_mat = self.create_dct_matrix(block_size)
        self.register_buffer("dct_mat", dct_mat)

        zigzag = build_zigzag_indices(block_size)

        # 去掉 DC，即 zigzag[0] = (0, 0)
        selected = zigzag[1:1 + num_coeff]

        self.selected_u = [p[0] for p in selected]
        self.selected_v = [p[1] for p in selected]

    def create_dct_matrix(self, N):
        mat = torch.zeros(N, N)

        for k in range(N):
            for n in range(N):
                if k == 0:
                    alpha = math.sqrt(1.0 / N)
                else:
                    alpha = math.sqrt(2.0 / N)

                mat[k, n] = alpha * math.cos(
                    math.pi * (2 * n + 1) * k / (2 * N)
                )

        return mat

    def normalize_per_channel(self, x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (x - mean) / std

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.block_size

        H8 = H // N * N
        W8 = W // N * N

        x = x[:, :, :H8, :W8]

        # [B, C, H/8, W/8, 8, 8]
        patches = x.unfold(2, N, N).unfold(3, N, N)

        D = self.dct_mat

        # DCT = D * block * D^T
        patches = torch.matmul(D, patches)
        patches = torch.matmul(patches, D.t())

        coeffs = []

        for u, v in zip(self.selected_u, self.selected_v):
            coeffs.append(patches[..., u, v])

        # [B, C, K, H/8, W/8]
        coeffs = torch.stack(coeffs, dim=2)

        coeffs = torch.log1p(torch.abs(coeffs))

        # [B, C*K, H/8, W/8]
        coeffs = coeffs.contiguous().view(
            B,
            C * self.num_coeff,
            H8 // N,
            W8 // N
        )

        coeffs = self.normalize_per_channel(coeffs)

        return coeffs


# =========================================================
# 7. Boundary Artifact Expert
# =========================================================

class BoundaryArtifactPreprocess(nn.Module):
    """
    作用：
        提取边界异常特征。

    组成：
        1. Sobel gradient
        2. Laplacian response
        3. gradient * laplacian combined residual

    主要用于：
        AE 类局部编辑、拼接、修补、擦除造成的边界不连续。
    """
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32)

        lap = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32)

        self.register_buffer("kx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("ky", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("klap", lap.view(1, 1, 3, 3))

    def normalize_per_channel(self, x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (x - mean) / std

    def forward(self, x):
        gray = (
            0.299 * x[:, 0:1] +
            0.587 * x[:, 1:2] +
            0.114 * x[:, 2:3]
        )

        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)

        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        lap = torch.abs(F.conv2d(gray, self.klap, padding=1))

        combined = grad * lap

        out = torch.cat([grad, lap, combined], dim=1)
        out = self.normalize_per_channel(out)

        return out


# =========================================================
# 8. 单个 Expert Branch
# =========================================================

class ExpertBranch(nn.Module):
    """
    单个 Expert 结构：

        input image
            ↓
        preprocess
            ↓
        ResNet18 backbone
            ↓
        projector
            ↓
        feature_dim 维特征

    aux_classifier：
        每个 Expert 都有一个辅助分类头。
        作用是防止某些 Expert 在训练中完全不学习。
    """
    def __init__(
        self,
        preprocess,
        in_channels,
        feature_dim=256,
        pretrained=False
    ):
        super().__init__()

        self.preprocess = preprocess

        self.backbone = ResNet18Feature(
            in_channels=in_channels,
            pretrained=pretrained and in_channels == 3
        )

        self.projector = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.aux_classifier = nn.Linear(feature_dim, 2)

    def forward(self, x):
        x_p = self.preprocess(x)

        feat = self.backbone(x_p)
        feat = self.projector(feat)

        aux_logits = self.aux_classifier(feat)

        return feat, aux_logits


# =========================================================
# 9. Feature-level MoE Router V2
# =========================================================

class FeatureMoERouter(nn.Module):
    """
    Feature-level MoE Router V2。

    五个专家：
        1. RGB Expert
           ImageNet Normalize + ResNet18
           负责语义、纹理、整体视觉异常。

        2. Noise Expert
           SRM residual + Bayar constrained conv
           参考 TruFor / Noiseprint
           负责噪声残差、相机指纹、局部编辑异常。

        3. FFT Expert
           FFT amplitude + high-frequency residual
           负责生成图频域异常。

        4. CAT-DCT Expert
           8x8 block DCT，去 DC，取 AC / 中高频系数
           参考 CAT-Net
           负责 JPEG/DCT 压缩痕迹、块效应、编辑区域压缩不一致。

        5. Boundary Expert
           Sobel + Laplacian + combined residual
           负责局部编辑边界异常。

    输出：
        logits:
            AE / DMG 分类结果。

        gate_weights:
            五个 Expert 的动态权重。

        aux_logits:
            每个 Expert 的辅助分类结果。

        feature_dict:
            如果 return_features=True，返回中间特征。
    """
    def __init__(self, feature_dim=256, pretrained_rgb=True):
        super().__init__()

        # 1. RGB Expert
        self.rgb_expert = ExpertBranch(
            preprocess=RGBNormalizePreprocess(),
            in_channels=3,
            feature_dim=feature_dim,
            pretrained=pretrained_rgb
        )

        # 2. Noise Expert
        # 5 个 SRM channel + 3 个 Bayar residual channel = 8 channel
        self.noise_expert = ExpertBranch(
            preprocess=TruForNoisePreprocess(),
            in_channels=8,
            feature_dim=feature_dim,
            pretrained=False
        )

        # 3. FFT Expert
        # 3 个 amplitude channel + 3 个 high-frequency residual channel = 6 channel
        self.fft_expert = ExpertBranch(
            preprocess=FFTPreprocess(),
            in_channels=6,
            feature_dim=feature_dim,
            pretrained=False
        )

        # 4. CAT-DCT Expert
        # 默认选 32 个 AC DCT 系数，RGB 三通道，所以输入通道数为 3*32=96
        self.dct_num_coeff = 32

        self.dct_expert = ExpertBranch(
            preprocess=CATDCTPreprocess(
                block_size=8,
                num_coeff=self.dct_num_coeff
            ),
            in_channels=3 * self.dct_num_coeff,
            feature_dim=feature_dim,
            pretrained=False
        )

        # 5. Boundary Expert
        # grad / lap / combined 三通道
        self.boundary_expert = ExpertBranch(
            preprocess=BoundaryArtifactPreprocess(),
            in_channels=3,
            feature_dim=feature_dim,
            pretrained=False
        )

        self.num_experts = 5

        self.gate = nn.Sequential(
            nn.Linear(feature_dim * self.num_experts, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_experts)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x, return_aux=True, return_features=False):
        f_rgb, aux_rgb = self.rgb_expert(x)
        f_noise, aux_noise = self.noise_expert(x)
        f_fft, aux_fft = self.fft_expert(x)
        f_dct, aux_dct = self.dct_expert(x)
        f_boundary, aux_boundary = self.boundary_expert(x)

        feats = [
            f_rgb,
            f_noise,
            f_fft,
            f_dct,
            f_boundary
        ]

        aux_logits = [
            aux_rgb,
            aux_noise,
            aux_fft,
            aux_dct,
            aux_boundary
        ]

        concat_feat = torch.cat(feats, dim=1)

        gate_logits = self.gate(concat_feat)
        gate_weights = torch.softmax(gate_logits, dim=1)

        stacked_feats = torch.stack(feats, dim=1)

        fused_feat = torch.sum(
            gate_weights.unsqueeze(-1) * stacked_feats,
            dim=1
        )

        logits = self.classifier(fused_feat)

        if return_features:
            feature_dict = {
                "f_rgb": f_rgb,
                "f_noise": f_noise,
                "f_fft": f_fft,
                "f_dct": f_dct,
                "f_boundary": f_boundary,
                "f_fused": fused_feat,
                "gate_weights": gate_weights,
                "logits": logits
            }

            if return_aux:
                feature_dict["aux_logits"] = aux_logits

            return feature_dict

        if return_aux:
            return logits, gate_weights, aux_logits
        else:
            return logits, gate_weights
