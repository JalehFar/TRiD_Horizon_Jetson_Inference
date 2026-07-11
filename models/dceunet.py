from __future__ import annotations

import torch
import torch.nn as nn


def build_norm_layer(norm_layer: dict, dim: int):
    norm_type = norm_layer.get("type", "BN") if norm_layer else "BN"
    if norm_type != "BN":
        raise ValueError(f"Only BatchNorm is supported in inference package, got {norm_type}")
    return "bn", nn.BatchNorm2d(dim)


class ECA(nn.Module):
    def __init__(self, channel: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, stride: int = 1, batch_norm: bool = True, preactivation: bool = False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = []
        if preactivation:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False))
            if batch_norm:
                layers = [nn.BatchNorm2d(in_channel)] + layers
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthWiseSeparateConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, stride: int = 1, batch_norm: bool = True, preactivation: bool = False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        if preactivation:
            layers: list[nn.Module] = [nn.ReLU(), nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=False), nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)]
            if batch_norm:
                layers = [nn.BatchNorm2d(in_channel)] + layers
        else:
            layers = [nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=False), nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DenseFeatureStack(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int, unit: int, growth_rate: int):
        super().__init__()
        self.conv_units = nn.ModuleList()
        current_in = in_channel
        for _ in range(unit):
            self.conv_units.append(ConvBlock(current_in, growth_rate, kernel_size, stride=1, batch_norm=True, preactivation=True))
            current_in += growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stack_feature = None
        for conv in self.conv_units:
            inputs = x if stack_feature is None else torch.cat([x, stack_feature], dim=1)
            out = conv(inputs)
            stack_feature = out if stack_feature is None else torch.cat([stack_feature, out], dim=1)
        return torch.cat([x, stack_feature], dim=1)


class GlobalContextFusion(nn.Module):
    def __init__(self, in_channels, max_pool_kernels, ch, ch_k, ch_v, br):
        super().__init__()
        self.ch_bottle = in_channels[-1]
        self.ch_in = ch * br
        self.ch = ch
        self.ch_k = ch_k
        self.ch_v = ch_v
        self.br = br
        self.ch_convs = nn.ModuleList([DepthWiseSeparateConvBlock(inc, ch, 3, 1, batch_norm=True, preactivation=True) for inc in in_channels])
        self.max_pool_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=k) for k in max_pool_kernels])
        self.ch_Wq = DepthWiseSeparateConvBlock(self.ch_in, self.ch_in, 1, 1, batch_norm=True, preactivation=True)
        self.ch_Wk = DepthWiseSeparateConvBlock(self.ch_in, 1, 1, 1, batch_norm=True, preactivation=True)
        self.ch_Wv = DepthWiseSeparateConvBlock(self.ch_in, self.ch_in, 1, 1, batch_norm=True, preactivation=True)
        self.ch_softmax = nn.Softmax(dim=1)
        self.ch_score_conv = nn.Conv2d(self.ch_in, self.ch_in, 1)
        self.ch_layer_norm = nn.LayerNorm((self.ch_in, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.sp_Wq = DepthWiseSeparateConvBlock(self.ch_in, br * ch_k, 1, 1, batch_norm=True, preactivation=True)
        self.sp_Wk = DepthWiseSeparateConvBlock(self.ch_in, br * ch_k, 1, 1, batch_norm=True, preactivation=True)
        self.sp_Wv = DepthWiseSeparateConvBlock(self.ch_in, br * ch_v, 1, 1, batch_norm=True, preactivation=True)
        self.sp_softmax = nn.Softmax(dim=-1)
        self.sp_output_conv = DepthWiseSeparateConvBlock(br * ch_v, self.ch_in, 1, 1, batch_norm=True, preactivation=True)
        self.output_conv = DepthWiseSeparateConvBlock(self.ch_in, self.ch_bottle, 3, 1, batch_norm=True, preactivation=True)

    def forward(self, feature_maps):
        max_pool_maps = [pool(f) for pool, f in zip(self.max_pool_layers, feature_maps)]
        ch_outs = [conv(m) for conv, m in zip(self.ch_convs, max_pool_maps)]
        x = torch.cat(ch_outs, dim=1)
        bs, _, h, w = x.size()
        ch_q = self.ch_Wq(x).reshape(bs, -1, h * w)
        ch_k = self.ch_softmax(self.ch_Wk(x).reshape(bs, -1, 1))
        z_ch = torch.matmul(ch_q, ch_k).unsqueeze(-1)
        ch_score = self.sigmoid(self.ch_layer_norm(self.ch_score_conv(z_ch)))
        ch_out = self.ch_Wv(x) * ch_score
        sp_q = self.sp_Wq(ch_out).reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1).reshape(bs, self.ch_k, -1)
        sp_k = self.sp_Wk(ch_out).reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1)
        sp_k = self.sp_softmax(sp_k.mean(-1).mean(-1).mean(-1).reshape(bs, 1, self.ch_k))
        z_sp = torch.matmul(sp_k, sp_q).reshape(bs, 1, h, w, self.br)
        sp_score = self.sigmoid(z_sp)
        sp_v = self.sp_Wv(ch_out).reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1)
        sp_out = (sp_v * sp_score).permute(0, 4, 1, 2, 3).reshape(bs, self.br * self.ch_v, h, w)
        return self.output_conv(self.sp_output_conv(sp_out))


class MultiScaleGaussian(nn.Module):
    def __init__(self, dim: int, sizes=(3, 5), sigmas=(0.8, 1.2)):
        super().__init__()
        self.filters = nn.ModuleList()
        for size, sigma in zip(sizes, sigmas):
            kernel = self._build_kernel(size, sigma)
            conv = nn.Conv2d(dim, dim, kernel_size=size, padding=size // 2, groups=dim, bias=False)
            conv.weight.data.copy_(kernel.repeat(dim, 1, 1, 1))
            conv.weight.requires_grad = False
            self.filters.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(f(x) for f in self.filters) / len(self.filters)

    @staticmethod
    def _build_kernel(size: int, sigma: float):
        kernel_range = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        xx, yy = torch.meshgrid(kernel_range, kernel_range, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)


class ScharrEdge(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32)
        scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
        self.conv_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.conv_y = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.conv_x.weight.data.copy_(scharr_x.view(1, 1, 3, 3).repeat(dim, 1, 1, 1))
        self.conv_y.weight.data.copy_(scharr_y.view(1, 1, 3, 3).repeat(dim, 1, 1, 1))
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)


class EGACore(nn.Module):
    def __init__(self, dim: int, norm_layer=dict(type="BN"), act_layer=nn.GELU):
        super().__init__()
        self.gaussian = MultiScaleGaussian(dim)
        self.scharr = ScharrEdge(dim)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        self.eca = ECA(dim)
        self.fuse_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), build_norm_layer(norm_layer, dim)[1], act_layer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.norm(self.act(x + self.gaussian(x) + self.scharr(x)))
        return self.fuse_conv(self.eca(fused))


class EGA(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, norm_layer=dict(type="BN"), act_layer=nn.GELU):
        super().__init__()
        self.channel_proj = nn.Conv2d(input_channels, num_classes, kernel_size=1)
        self.enhancer = EGACore(num_classes, norm_layer=norm_layer, act_layer=act_layer)
        self.shortcut = nn.Conv2d(input_channels, num_classes, kernel_size=1) if input_channels != num_classes else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enhancer(self.channel_proj(x)) + self.shortcut(x)


class DownSampleEGA(nn.Module):
    def __init__(self, in_channel, base_channel, kernel_size, unit, growth_rate, skip_channel=None, downsample=True, skip=True):
        super().__init__()
        self.skip = skip
        stride = 2 if downsample else 1
        self.downsampler = ConvBlock(in_channel, in_channel, 3, stride=stride, batch_norm=True, preactivation=True)
        self.ega_enhancer = EGA(in_channel, base_channel)
        self.dense_stack = DenseFeatureStack(base_channel, 3, unit, growth_rate)
        if skip:
            self.skip_conv = ConvBlock(base_channel + unit * growth_rate, skip_channel, 3, stride=1, batch_norm=True, preactivation=True)

    def forward(self, x: torch.Tensor):
        x = self.dense_stack(self.ega_enhancer(self.downsampler(x)))
        if self.skip:
            return x, self.skip_conv(x)
        return x


class DCEU(nn.Module):
    def __init__(self, inp: int, oup: int, groups: int = 32, reduction: int = 4, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))
        mip = max(8, inp // groups)
        self.shared_conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.shared_bn = nn.BatchNorm2d(mip)
        self.relu = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(inp, inp // reduction, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(inp // reduction, 2, 1), nn.Softmax(dim=1))
        self.channel_att = ECA(oup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, h, w = x.size()
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = self.relu(self.shared_bn(self.shared_conv1(torch.cat([x_h_mean, x_w_mean], dim=2))))
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        attn_mean = self.conv_h(x_h_mean).sigmoid() * self.conv_w(x_w_mean.permute(0, 1, 3, 2)).sigmoid()
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = self.relu(self.shared_bn(self.shared_conv1(torch.cat([x_h_max, x_w_max], dim=2))))
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        attn_max = self.conv_h(x_h_max).sigmoid() * self.conv_w(x_w_max.permute(0, 1, 3, 2)).sigmoid()
        weights = self.gate(identity)
        out = self.channel_att(identity * (attn_mean * weights[:, 0:1] + attn_max * weights[:, 1:2]))
        return out + identity if self.use_residual else out


class DCEUNet(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 1):
        super().__init__()
        base_channels = [24, 24, 24]
        skip_channels = [12, 24, 24]
        units = [3, 5, 5]
        kernel_sizes = [5, 3, 3]
        growth_rates = [4, 8, 16]
        downsample_channels = [base_channels[i] + units[i] * growth_rates[i] for i in range(len(base_channels))]
        self.down_convs = nn.ModuleList()
        for i in range(3):
            self.down_convs.append(DownSampleEGA(input_channels if i == 0 else downsample_channels[i - 1], base_channels[i], kernel_sizes[i], units[i], growth_rates[i], skip_channels[i], True, True))
        self.global_fusion = GlobalContextFusion(downsample_channels, [4, 2, 1], ch=48, ch_k=48, ch_v=48, br=3)
        self.dceu_enhancer = DCEU(downsample_channels[-1], downsample_channels[-1])
        self.bottle_conv = ConvBlock(downsample_channels[2] + skip_channels[2], skip_channels[2], 3, stride=1, batch_norm=True, preactivation=True)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_2 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.out_conv = ConvBlock(sum(skip_channels), num_classes, 3, stride=1, batch_norm=True, preactivation=True)
        self.upsample_out = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x1, skip1 = self.down_convs[0](x)
        x2, skip2 = self.down_convs[1](x1)
        x3, skip3 = self.down_convs[2](x2)
        fused = self.global_fusion([x1, x2, x3])
        enhanced = self.dceu_enhancer(fused)
        bottle = self.bottle_conv(torch.cat([enhanced, skip3], dim=1))
        skip2u = self.upsample_1(skip2)
        skip3u = self.upsample_2(bottle)
        logits_128 = self.out_conv(torch.cat([skip1, skip2u, skip3u], dim=1))
        seg_logits = self.upsample_out(logits_128)
        return {"x1": x1, "skip1": skip1, "x2": x2, "skip2": skip2, "x3": x3, "skip3": skip3, "fused": fused, "enhanced": enhanced, "bottle": bottle, "logits_128": logits_128, "seg_logits": seg_logits}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["seg_logits"]
