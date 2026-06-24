# moe_router_model_v3.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utils
# =========================================================

def rgb_to_gray(x):
    # x: [B,3,H,W]
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def safe_log1p_abs(x):
    return torch.log1p(torch.abs(x)) * torch.sign(x)


def make_conv_kernel(kernel, device, dtype):
    k = torch.tensor(kernel, device=device, dtype=dtype).view(1, 1, len(kernel), len(kernel[0]))
    return k


def apply_depthwise_kernel(x, kernel):
    # x: [B,C,H,W], kernel: [1,1,kh,kw]
    b, c, h, w = x.shape
    k = kernel.repeat(c, 1, 1, 1)
    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2
    return F.conv2d(x, k, padding=(pad_h, pad_w), groups=c)


# =========================================================
# 1. Enhanced NPR Preprocess
# =========================================================

class EnhancedNPRPreprocess(nn.Module):
    """
    Multi-order neighboring pixel relationship.
    Output channels: 7 * 3 = 21
    dx, dy, diag1, diag2, dxx, dyy, local mean residual.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x normalized or tensor in roughly image range.
        # Keep residual scale stable.
        dx = x - torch.roll(x, shifts=1, dims=3)
        dy = x - torch.roll(x, shifts=1, dims=2)
        dxy1 = x - torch.roll(torch.roll(x, shifts=1, dims=2), shifts=1, dims=3)
        dxy2 = x - torch.roll(torch.roll(x, shifts=1, dims=2), shifts=-1, dims=3)

        dxx = x - 2 * torch.roll(x, shifts=1, dims=3) + torch.roll(x, shifts=2, dims=3)
        dyy = x - 2 * torch.roll(x, shifts=1, dims=2) + torch.roll(x, shifts=2, dims=2)

        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        pred_res = x - local_mean

        out = torch.cat([dx, dy, dxy1, dxy2, dxx, dyy, pred_res], dim=1)
        return safe_log1p_abs(out)


# =========================================================
# 2. Enhanced Forensic Noise Preprocess
# =========================================================

class EnhancedForensicNoisePreprocess(nn.Module):
    """
    SRM-like residual + prediction residual + local variance.
    Output channels:
      SRM 5 kernels * 3 = 15
      local variance gray = 1
      prediction residual RGB = 3
    Total = 19
    """

    def __init__(self):
        super().__init__()

        srm_kernels = []

        srm_kernels.append([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ])

        srm_kernels.append([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ])

        srm_kernels.append([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])

        srm_kernels.append([
            [0, 0, 1, 0, 0],
            [0, 0, -2, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])

        srm_kernels.append([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ])

        weight = torch.tensor(srm_kernels, dtype=torch.float32).unsqueeze(1)
        self.register_buffer("srm_weight", weight)

    def forward(self, x):
        b, c, h, w = x.shape

        # SRM residual per RGB channel.
        srm_feats = []
        for ch in range(3):
            xi = x[:, ch:ch + 1]
            ri = F.conv2d(xi, self.srm_weight, padding=2)
            srm_feats.append(ri)
        srm = torch.cat(srm_feats, dim=1)

        # Local variance on gray.
        gray = rgb_to_gray(x)
        mean = F.avg_pool2d(gray, 5, stride=1, padding=2)
        mean2 = F.avg_pool2d(gray * gray, 5, stride=1, padding=2)
        var = torch.clamp(mean2 - mean * mean, min=0.0)

        # Local prediction residual.
        pred = F.avg_pool2d(x, 3, stride=1, padding=1)
        pred_res = x - pred

        out = torch.cat([srm, var, pred_res], dim=1)
        return safe_log1p_abs(out)


# =========================================================
# 3. Enhanced Spectrum Preprocess
# =========================================================

class EnhancedSpectrumPreprocess(nn.Module):
    """
    FFT log amplitude + high-frequency residual + radial emphasis.
    Output channels: 9
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        fft = torch.fft.fft2(x, norm="ortho")
        amp = torch.log1p(torch.abs(fft))
        amp = torch.fft.fftshift(amp, dim=(-2, -1))

        # Low frequency smoothed amplitude.
        low = F.avg_pool2d(amp, kernel_size=9, stride=1, padding=4)
        high_res = amp - low

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
        rr = torch.sqrt(xx * xx + yy * yy).clamp(0, 1).view(1, 1, h, w)
        radial = amp * rr

        out = torch.cat([amp, high_res, radial], dim=1)
        return out


# =========================================================
# 4. Enhanced DCT-JPEG Preprocess
# =========================================================

class EnhancedDCTJPEGPreprocess(nn.Module):
    """
    Block 8x8 DCT coefficient maps + block boundary artifact.
    Default num_coeff=32.
    Output channels: 3 * num_coeff + 1.
    """

    def __init__(self, num_coeff=32):
        super().__init__()
        self.num_coeff = num_coeff
        dct = self._make_dct_filters(num_coeff)
        self.register_buffer("dct_weight", dct)

    def _make_dct_filters(self, num_coeff):
        filters = []
        coeffs = []
        for u in range(8):
            for v in range(8):
                if u == 0 and v == 0:
                    continue
                freq = u + v
                coeffs.append((freq, u, v))
        coeffs = sorted(coeffs, key=lambda t: t[0])[:num_coeff]

        for _, u, v in coeffs:
            basis = torch.zeros(8, 8)
            alpha_u = math.sqrt(1 / 8) if u == 0 else math.sqrt(2 / 8)
            alpha_v = math.sqrt(1 / 8) if v == 0 else math.sqrt(2 / 8)
            for x in range(8):
                for y in range(8):
                    basis[x, y] = alpha_u * alpha_v * \
                                  math.cos(((2 * x + 1) * u * math.pi) / 16) * \
                                  math.cos(((2 * y + 1) * v * math.pi) / 16)
            filters.append(basis)
        weight = torch.stack(filters, dim=0).unsqueeze(1)
        return weight.float()

    def forward(self, x):
        b, c, h, w = x.shape

        # Pad to multiple of 8.
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        xp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        feats = []
        for ch in range(3):
            xi = xp[:, ch:ch + 1]
            ci = F.conv2d(xi, self.dct_weight, stride=8)
            ci = torch.log1p(torch.abs(ci))
            ci = F.interpolate(ci, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(ci)

        dct_feats = torch.cat(feats, dim=1)

        # Block boundary artifact on gray.
        gray = rgb_to_gray(x)
        vdiff = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        hdiff = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

        vmap = torch.zeros_like(gray)
        hmap = torch.zeros_like(gray)
        vmap[:, :, :, 1:] = vdiff
        hmap[:, :, 1:, :] = hdiff

        mask_v = torch.zeros_like(gray)
        mask_h = torch.zeros_like(gray)
        mask_v[:, :, :, 7::8] = 1.0
        mask_h[:, :, 7::8, :] = 1.0
        block_artifact = vmap * mask_v + hmap * mask_h

        out = torch.cat([dct_feats, block_artifact], dim=1)
        return out


# =========================================================
# 5. Boundary-HF Preprocess
# =========================================================

class BoundaryHFPreprocess(nn.Module):
    """
    Sobel + Laplacian + high-pass + interaction + local contrast.
    Output channels:
      sobel_x RGB 3
      sobel_y RGB 3
      lap RGB 3
      highpass RGB 3
      interaction gray 1
      local contrast gray 1
    Total = 14
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        device, dtype = x.device, x.dtype

        sobel_x = make_conv_kernel([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], device, dtype)

        sobel_y = make_conv_kernel([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ], device, dtype)

        lap = make_conv_kernel([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ], device, dtype)

        gx = apply_depthwise_kernel(x, sobel_x)
        gy = apply_depthwise_kernel(x, sobel_y)
        lp = apply_depthwise_kernel(x, lap)

        blur = F.avg_pool2d(x, 5, stride=1, padding=2)
        hp = x - blur

        gray = rgb_to_gray(x)
        gxg = apply_depthwise_kernel(gray, sobel_x)
        gyg = apply_depthwise_kernel(gray, sobel_y)
        lpg = apply_depthwise_kernel(gray, lap)
        grad_mag = torch.sqrt(gxg * gxg + gyg * gyg + 1e-6)
        interaction = grad_mag * torch.abs(lpg)

        mean = F.avg_pool2d(gray, 5, stride=1, padding=2)
        mean2 = F.avg_pool2d(gray * gray, 5, stride=1, padding=2)
        contrast = torch.sqrt(torch.clamp(mean2 - mean * mean, min=0.0) + 1e-6)

        out = torch.cat([gx, gy, lp, hp, interaction, contrast], dim=1)
        return safe_log1p_abs(out)


# =========================================================
# 6. Reconstruction Residual Preprocess
# =========================================================

class ReconResidualPreprocess(nn.Module):
    """
    Lightweight DIRE-inspired residual:
      gaussian/avg blur residual
      down-up residual
      avg lowpass residual
      median-like approximate by max/min pooling residual
    Output channels: 12
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        blur = F.avg_pool2d(x, 5, stride=1, padding=2)
        res_blur = x - blur

        down = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False)
        res_downup = x - up

        low = F.avg_pool2d(x, 9, stride=1, padding=4)
        res_low = x - low

        maxp = F.max_pool2d(x, 3, stride=1, padding=1)
        minp = -F.max_pool2d(-x, 3, stride=1, padding=1)
        mid = 0.5 * (maxp + minp)
        res_mid = x - mid

        out = torch.cat([res_blur, res_downup, res_low, res_mid], dim=1)
        return safe_log1p_abs(out)


# =========================================================
# Expert Backbone
# =========================================================

class SmallExpertCNN(nn.Module):
    def __init__(self, in_channels, feature_dim=256, width=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),

            nn.Conv2d(width, width, 3, stride=1, padding=1, groups=width, bias=False),
            nn.Conv2d(width, width * 2, 1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),

            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),

            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),

            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        return self.fc(self.net(x))


# =========================================================
# Feature MoE Router V3
# =========================================================

class FeatureMoERouterV3(nn.Module):
    def __init__(
        self,
        num_classes=2,
        feature_dim=256,
        use_rgb_expert=False,
        dct_num_coeff=32,
        expert_width=48,
        dropout=0.25,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_rgb_expert = use_rgb_expert
        self.dct_num_coeff = dct_num_coeff

        self.expert_names = [
            "NPR",
            "NOISE",
            "SPECTRUM",
            "DCT_JPEG",
            "BOUNDARY_HF",
            "RECON",
        ]

        self.pre_npr = EnhancedNPRPreprocess()
        self.pre_noise = EnhancedForensicNoisePreprocess()
        self.pre_spectrum = EnhancedSpectrumPreprocess()
        self.pre_dct = EnhancedDCTJPEGPreprocess(num_coeff=dct_num_coeff)
        self.pre_boundary = BoundaryHFPreprocess()
        self.pre_recon = ReconResidualPreprocess()

        self.exp_npr = SmallExpertCNN(21, feature_dim, expert_width)
        self.exp_noise = SmallExpertCNN(19, feature_dim, expert_width)
        self.exp_spectrum = SmallExpertCNN(9, feature_dim, expert_width)
        self.exp_dct = SmallExpertCNN(3 * dct_num_coeff + 1, feature_dim, expert_width)
        self.exp_boundary = SmallExpertCNN(14, feature_dim, expert_width)
        self.exp_recon = SmallExpertCNN(12, feature_dim, expert_width)

        experts = 6

        if use_rgb_expert:
            self.expert_names = ["RGB"] + self.expert_names
            self.exp_rgb = SmallExpertCNN(3, feature_dim, expert_width)
            experts += 1
        else:
            self.exp_rgb = None

        self.num_experts = experts

        self.router = nn.Sequential(
            nn.Linear(feature_dim * self.num_experts, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, self.num_experts),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        self.aux_heads = nn.ModuleList([
            nn.Linear(feature_dim, num_classes) for _ in range(self.num_experts)
        ])

    def forward(self, x, return_features=False):
        feats = []

        if self.use_rgb_expert:
            f_rgb = self.exp_rgb(x)
            feats.append(f_rgb)

        f_npr = self.exp_npr(self.pre_npr(x))
        f_noise = self.exp_noise(self.pre_noise(x))
        f_spectrum = self.exp_spectrum(self.pre_spectrum(x))
        f_dct = self.exp_dct(self.pre_dct(x))
        f_boundary = self.exp_boundary(self.pre_boundary(x))
        f_recon = self.exp_recon(self.pre_recon(x))

        feats.extend([f_npr, f_noise, f_spectrum, f_dct, f_boundary, f_recon])

        feat_stack = torch.stack(feats, dim=1)  # [B,E,D]
        router_in = torch.cat(feats, dim=1)
        gate_logits = self.router(router_in)
        gates = torch.softmax(gate_logits, dim=1)  # [B,E]

        fused = torch.sum(feat_stack * gates.unsqueeze(-1), dim=1)
        logits = self.classifier(fused)

        aux_logits = [head(feats[i]) for i, head in enumerate(self.aux_heads)]

        if return_features:
            return {
                "logits": logits,
                "aux_logits": aux_logits,
                "gates": gates,
                "features": {
                    name: feat_stack[:, i, :] for i, name in enumerate(self.expert_names)
                },
                "fused": fused,
            }

        return logits, aux_logits, gates


def build_model_v3(
    num_classes=2,
    feature_dim=256,
    use_rgb_expert=False,
    dct_num_coeff=32,
    expert_width=48,
    dropout=0.25,
):
    return FeatureMoERouterV3(
        num_classes=num_classes,
        feature_dim=feature_dim,
        use_rgb_expert=use_rgb_expert,
        dct_num_coeff=dct_num_coeff,
        expert_width=expert_width,
        dropout=dropout,
    )
