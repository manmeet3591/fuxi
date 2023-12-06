import numpy as np
import pandas as pd
import collections.abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

from functools import partial
from einops import repeat, rearrange

__all__ = ['FuXi', 'UTransformer', 'time_encoding']


def exists(val):
    return val is not None


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def time_encoding(init_time, num_steps, freq=6):
    ''' 
    This function generates time encodings for the given initial time, 
    number of steps, and frequency of time intervals. The approach is 
    based on the description provided in the paper 
    "GraphCast: Learning skillful medium-range global weather forecasting" 
    by DeepMind (https://arxiv.org/abs/2212.12794). 
    '''

    init_time = np.array([init_time])
    tembs = []
    for i in range(num_steps):
        hours = np.array([pd.Timedelta(hours=t*freq) for t in [i-1, i, i+1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, 'H') for t in times.reshape(-1)]
        times = [(p.day_of_year/366, p.hour/24) for p in times]
        temb = torch.from_numpy(np.array(times, dtype=np.float32))
        temb = torch.cat([temb.sin(), temb.cos()], dim=-1)
        temb = temb.reshape(1, -1)
        tembs.append(temb)
    return torch.stack(tembs)


class CubeEmbed(nn.Module):
    def __init__(
        self,
        in_chans=3,
        in_frames=1,
        patch_size=4,
        embed_dim=96,
        norm_layer=None,
        flatten=False,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans * in_frames, embed_dim,
            kernel_size=(patch_size[0], patch_size[1]),
            stride=(patch_size[0], patch_size[1]),
            padding=0
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if x.ndim == 5:
            x = rearrange(x, 'n t c h w -> n (t c) h w')

        x = self.proj(x)

        if self.flatten:
            x = rearrange(x, 'n c h w -> n (h w) c')
            x = self.norm(x)

        return x


class PeriodicConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert max(self.padding) > 0

    def forward(self, x):
        x = F.pad(x, (self.padding[1], self.padding[1], 0, 0), mode="circular")
        x = F.pad(
            x, (0, 0, self.padding[0], self.padding[0]), mode="constant", value=0)
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups
        )
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        temb_dim=0,
        dropout=0,
        scale_shift=False,
        conv_class=nn.Conv2d,
    ):
        super().__init__()

        self.scale_shift = scale_shift

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_dim,
                         eps=1e-6, affine=True),
            nn.SiLU(),
            conv_class(in_dim, out_dim, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, 2 * out_dim if scale_shift else out_dim)
        ) if temb_dim > 0 else nn.Identity()

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_dim,
                         eps=1e-6, affine=True),
            nn.SiLU() if scale_shift else nn.Identity(),
            nn.Dropout(p=dropout),
            conv_class(out_dim, out_dim, 3, padding=1)
        )

        if in_dim == out_dim:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x, temb=None):
        h = self.in_layers(x)
        if temb is None:
            h = self.out_layers(h)
        else:
            temb = self.emb_layers(temb)
            temb = rearrange(temb, 'n c -> n c 1 1')
            if self.scale_shift:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = temb.chunk(2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + temb
                h = self.out_layers(h)
        return self.skip_connection(x) + h


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, use_conv=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2

        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(
        self,
        channels,
        out_channels=None,
        use_conv=False,
        use_deconv=False,
        conv_class=nn.Conv2d
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_deconv = use_deconv

        if use_deconv:
            self.op = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.op = conv_class(
                self.channels, self.out_channels, 3, padding=1)

    def forward(self, x, output_size=None):
        if self.use_deconv:
            return self.op(x)

        if output_size is None:
            y = F.interpolate(x.float(), scale_factor=2.0,
                              mode="nearest").to(x)
        else:
            y = F.interpolate(x.float(), size=tuple(
                output_size), mode="nearest").to(x)

        if self.use_conv:
            y = self.op(y)

        return y


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int = 0,
        num_layers: int = 1,
        down=True,
        scale_shift=False,
        conv_class=nn.Conv2d,
    ):
        super().__init__()
        self.down = down

        if down:
            self.downsample = Downsample(
                in_channels, out_channels, use_conv=True)

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                ResBlock(
                    out_channels if down else in_channels,
                    out_channels,
                    temb_dim=temb_dim,
                    scale_shift=scale_shift,
                    conv_class=conv_class,
                ),
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x, temb=None):

        if self.down:
            x = self.downsample(x)

        for blk in self.resnets:
            x = blk(x, temb)
        return x


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        temb_dim: int = 0,
        num_layers: int = 1,
        up=True,
        scale_shift=False,
        conv_class=nn.Conv2d,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels + skip_channels if i == 0 else out_channels
            resnets.append(
                ResBlock(
                    in_channels,
                    out_channels,
                    temb_dim=temb_dim,
                    scale_shift=scale_shift,
                    conv_class=conv_class,
                ),
            )
        self.resnets = nn.ModuleList(resnets)

        self.up = up
        if up:
            self.upsample = Upsample(
                out_channels, out_channels, use_deconv=True)

    def forward(self, x, temb=None, output_size=None):
        for blk in self.resnets:
            x = blk(x, temb)

        if self.up:
            x = self.upsample(x, output_size)
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class GeGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        inner_dim = int(hidden_features * (2 / 3))
        self.fc1 = nn.Linear(in_features, inner_dim * 2, bias=False)
        self.act = GEGLU()
        self.fc2 = nn.Linear(inner_dim, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0],
               W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttentionV2(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        pretrained_window_size=[0, 0]
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)

        relative_coords_table = torch.stack(
            torch.meshgrid(
                [relative_coords_h, relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :,
                                  0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :,
                                  1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(
                self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @
                F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(
            torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table.to(x)).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.to(x)
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        input_size,
        window_size=7,
        shift_size=0,
        mask_type='h',
        mlp_ratio=4.,
        qkv_bias=True,
        drop=0.,
        attn_drop=0.,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.input_size = input_size
        self.num_heads = num_heads
        self.window_size = list(to_2tuple(window_size))
        self.shift_size = list(to_2tuple(shift_size))
        self.mlp_ratio = mlp_ratio

        if self.input_size[0] <= self.window_size[0]:
            self.shift_size[0] = 0
            self.window_size[0] = self.input_size[0]

        if self.input_size[1] <= self.window_size[1]:
            self.shift_size[1] = 0
            self.window_size[1] = self.input_size[1]

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttentionV2(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GeGLUFFN(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if max(self.shift_size) > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_size
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    if mask_type == 'h':
                        img_mask[:, h, :, :] = cnt
                    elif mask_type == 'w':
                        img_mask[:, :, w, :] = cnt
                    else:
                        img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        # cyclic shift
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1,
                                   self.window_size[0] * self.window_size[1], C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


conv_class = PeriodicConv2d
norm_layer = partial(nn.LayerNorm, eps=1e-6)


class SwinLayer(nn.Module):

    def __init__(
        self,
        in_chans,
        embed_dim,
        input_size,
        window_size,
        depth=4,
        num_heads=8,
        mlp_ratio=4.,
    ):

        super().__init__()

        self.depth = depth
        self.input_size = input_size

        self.blocks = nn.ModuleList()

        for i in range(depth):
            blk = SwinBlock(
                dim=embed_dim,
                input_size=input_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
            self.blocks.append(blk)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, h):
        for i, blk in enumerate(self.blocks):
            h = blk(h)
        return h


class UTransformer(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        in_frames,
        image_size,
        window_size=8,
        patch_size=4,
        down_times=0,
        embed_dim=1024,
        num_heads=8,
        depths=[6, 6, 6, 6],
        mlp_ratio=4,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.feat_size = [sz // patch_size for sz in image_size]

        self.num_layers = len(depths)
        self.down_times = down_times

        self.patch_embed = CubeEmbed(
            in_chans=in_chans,
            in_frames=in_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=True,
        )

        self.time_embed = nn.Sequential(
            nn.Linear(12, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        if self.down_times > 0:
            down_blocks = []
            up_blocks = []
            for i in range(self.down_times):
                down_blocks.append(
                    DownBlock2D(
                        embed_dim,
                        embed_dim,
                        temb_dim=embed_dim,
                        num_layers=min(4, (i+1)**2),
                        scale_shift=True,
                        conv_class=conv_class
                    )
                )
                up_blocks.append(
                    UpBlock2D(
                        embed_dim,
                        embed_dim,
                        embed_dim,
                        temb_dim=embed_dim,
                        num_layers=min(4, (self.down_times-i)**2),
                        scale_shift=True,
                        conv_class=conv_class
                    ),
                )

            self.down_blocks = nn.ModuleList(down_blocks)
            self.up_blocks = nn.ModuleList(up_blocks)

        layers = []
        input_size = [sz // (patch_size*(2 ** down_times))
                      for sz in image_size]

        for i in range(self.num_layers):
            layer = SwinLayer(
                embed_dim,
                embed_dim,
                input_size,
                window_size,
                depth=depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            layers.append(layer)
            self.add_module(f"norm{i}", norm_layer(embed_dim))

        self.layers = nn.ModuleList(layers)

        self.fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_layers, embed_dim),
            nn.GELU(),
        )

        self.head = nn.Linear(embed_dim, out_chans*patch_size*patch_size)

    def forward(self, input, temb=None, const=None, noise=None):

        if exists(const):
            x = torch.cat([input, const], dim=2)
        else:
            x = input

        x = rearrange(x, 'n t c h w -> n (t c) h w')
        h = self.patch_embed(x)

        if exists(noise):
            noise = F.interpolate(
                noise,
                size=self.feat_size,
                mode="bilinear",
                align_corners=False,
            )
            noise = rearrange(noise, 'n c h w -> n (h w) c')
            h = h + noise.to(h)

        if exists(temb):
            temb = self.time_embed(temb.to(h))

        # unet down
        if self.down_times > 0:
            h = rearrange(h, 'n (h w) c -> n c h w', h=self.feat_size[0])
            hs = []
            for blk in self.down_blocks:
                h = blk(h, temb)
                hs.append(h)
            feat_size = h.shape[-2:]
            h = rearrange(h, 'n c h w -> n (h w) c')

        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.layers):
            h = blk(h)
            out = getattr(self, f"norm{i}")(h)
            outs.append(out)
        h = self.fpn(torch.cat(outs, dim=-1))

        # unet up
        if self.down_times > 0:
            h = rearrange(h, 'n (h w) c -> n c h w', h=feat_size[0])
            for blk in self.up_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = blk(h, temb)
            h = rearrange(h, 'n c h w -> n (h w) c')

        out = self.head(h)
        out = rearrange(out,
                        'n (h w) (p1 p2 c) -> n c (h p1) (w p2)',
                        h=self.feat_size[0],
                        p1=self.patch_size,
                        p2=self.patch_size,
                        )

        if out.shape[-2:] != input.shape[-2:]:
            out = F.interpolate(
                out.float(),
                size=input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(input)

        return out


class FuXi(nn.Module):

    def __init__(
        self,
        in_frames,
        out_frames,
        step_range,
        decoder,
        device='cuda',
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.step_range = step_range
        self.decoder = nn.ModuleList(decoder)
        self.device = device
        self.dtype = dtype

    def load(self, model_dir, fmt='pth'):
        import os
        from safetensors import safe_open
        from safetensors.torch import save_file
        
        for i, name in enumerate(['fuxi_short', 'fuxi_medium', 'fuxi_long']):
            checkpoint = os.path.join(model_dir, f'{name}.{fmt}')
            print(f'load from {checkpoint} ...')
            
            model_state = {}
            if fmt == "pth":
                chkpt = torch.load(checkpoint, map_location=torch.device("cpu"))
                for k, v in chkpt["model"].items():
                    k = k.replace('decoder.', '')
                    model_state[k] = v

                save_name = os.path.join(model_dir, f'{name}.st')
                print(f'save to {save_name} ...')
                save_file(model_state, save_name)
                                        
                
            elif fmt == "st":
                with safe_open(checkpoint, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        model_state[k] = f.get_tensor(k)
            
            self.decoder[i].load_state_dict(model_state, strict=False)
        

        buffer = os.path.join(model_dir, f'buffer.{fmt}')
        if os.path.exists(buffer):
            print(f'load from {buffer} ...')
            if fmt == 'pth':
                buffer = torch.load(buffer)
                for k, v in buffer.items():
                    self.register_buffer(k, v.to(self.device))
            elif fmt == 'st':
                with safe_open(buffer, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        v = f.get_tensor(k)
                        self.register_buffer(k, v.to(self.device))


    def normalize(self, x, inv=False):
        mean = self.mean
        std = self.std

        if inv:
            x = x * std + mean
            tp = x[:, -1].exp() - 1
            x[:, -1] = tp.clamp(min=0)
        else:
            tp = x[:, -1]
            tp = torch.log(1 + tp.clamp(min=0))
            x[:, -1] = tp
            x = (x - mean) / std

        return x

    def process_input(self, input, hw=(720, 1440)):
        if input.shape[-2:] != hw:
            input = F.interpolate(
                input,
                size=hw,
                mode="bilinear",
                align_corners=False
            )

        input = self.normalize(input)
        # assert input.abs().max() < 60, (input.min().item(), input.max().item())

        const = repeat(self.const,
                       'c h w -> t c h w', t=input.size(0)
                       )
        
        input = input.to(self.dtype)
        const = const.to(self.dtype)
        return input[None], const[None]


    def process_output(self, output, hw=(721, 1440)):
        output = F.interpolate(
            output.float(),
            size=hw,
            mode="bilinear",
            align_corners=False
        )
        output = self.normalize(output, inv=True)
        return output

    @torch.no_grad()
    def forward(self, inputs):
        input, tembs = inputs
        num_steps = sum(self.step_range)
        outputs = input.new_zeros(1, num_steps, *input.shape[-3:])
        
        input, const = self.process_input(input, hw=(720, 1440))

        step = 0
        for i, future_frames in enumerate(self.step_range):
            decoder = self.decoder[i]
            for _ in range(0, future_frames):

                with autocast(dtype=self.dtype):
                    output = decoder(input, temb=tembs[step], const=const)

                output = rearrange(
                    output, 'n (t c) h w -> n t c h w', t=self.out_frames)
                output = output + input[:, -1:]

                print(f'stage: {i}, step: {1+step:02d}')

                outputs[:, step] = self.process_output(
                    output[0], hw=outputs.shape[-2:])

                input = torch.cat([input, output], dim=1)[:, -self.in_frames:]
                step += 1

            if step > num_steps:
                break

        return outputs
