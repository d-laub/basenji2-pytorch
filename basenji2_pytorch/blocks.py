from typing import Dict, Type

import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

BLOCK_REGISTRY: Dict[str, Type[nn.Module]] = {}

ACTIVATION_LAYER_REGISTRY: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}

KERAS_LAYER_DEFAULTS = {
    "BatchNorm1d": {
        "momentum": 1 - 0.99,  # momentum is 1-momentum
        "eps": 1e-3,
    },
}


class BasenjiSoftplus(nn.Module):
    def __init__(self, exp_max=10000):
        super().__init__()
        self.exp_max = exp_max

    def forward(self, x: torch.Tensor):
        x.clip_(-self.exp_max, self.exp_max)
        return F.softplus(x)


ACTIVATION_LAYER_REGISTRY["softplus"] = BasenjiSoftplus


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x


class Cropping1d(nn.Module):
    def __init__(self, cropping: int, **kwargs):
        super().__init__()
        self.cropping = cropping

    def forward(self, x):
        return x[..., self.cropping : -self.cropping]


BLOCK_REGISTRY["Cropping1D"] = Cropping1d


class KerasMaxPool1d(nn.Module):
    def __init__(
        self,
        pool_size=2,
        padding="valid",
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        """Hard coded to only be compatible with kernel size and stride of 2."""
        super().__init__()
        self.padding = padding
        _padding = 0
        if pool_size != 2:
            raise NotImplementedError("MaxPool1D with kernel size other than 2.")
        self.pool = nn.MaxPool1d(
            kernel_size=pool_size,
            padding=_padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, x):
        # (b c h)
        if self.padding == "same" and x.shape[-1] % 2 == 1:
            x = F.pad(x, (0, 1), value=-float("inf"))
        return self.pool(x)


class BasenjiConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size=1,
        activation="relu",
        stride=1,
        dilation_rate=1,
        dropout=0,
        pool_size=1,
        norm_type=None,
        bn_momentum=0.99,
        residual=False,
        padding="same",
    ):
        super().__init__()

        block = nn.ModuleList()

        # activation
        block.append(ACTIVATION_LAYER_REGISTRY[activation]())

        # conv
        block.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=int(round(dilation_rate)),
                bias=(norm_type is None),
            )
        )
        # batch norm
        if norm_type == "batch":
            # Use Keras default eps
            block.append(nn.BatchNorm1d(filters, momentum=1 - bn_momentum, eps=1e-3))
        else:
            raise NotImplementedError(f"BasenjiConvBlock with norm_type = {norm_type}")
        # dropout
        if dropout > 0:
            block.append(nn.Dropout(p=dropout))
        # residual
        if residual:
            block = nn.ModuleList([Residual(nn.Sequential(*block))])
        # pool
        if pool_size > 1:
            block.append(KerasMaxPool1d(pool_size, padding))

        self.block = nn.Sequential(*block)
        self.out_channels = filters

    def forward(self, x):
        return self.block(x)


BLOCK_REGISTRY["conv_block"] = BasenjiConvBlock


class BasenjiConvTower(nn.Module):
    def __init__(
        self,
        in_channels,
        filters_init,
        filters_end=None,
        filters_mult=None,
        divisible_by=1,
        repeat=1,
        **kwargs,
    ):
        super().__init__()

        def _round(x):
            return int(np.round(x / divisible_by) * divisible_by)

        # determine multiplier
        if filters_mult is None:
            assert filters_end is not None
            filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

        rep_filters = filters_init
        in_channels = in_channels
        tower = nn.ModuleList()
        for _ in range(repeat):
            tower.append(
                BasenjiConvBlock(
                    in_channels=in_channels, filters=_round(rep_filters), **kwargs
                )
            )
            in_channels = _round(rep_filters)
            rep_filters *= filters_mult

        self.tower = nn.Sequential(*tower)
        self.out_channels = in_channels

    def forward(self, x):
        return self.tower(x)


BLOCK_REGISTRY["conv_tower"] = BasenjiConvTower


class BasenjiDilatedResidual(nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size=3,
        rate_mult=2,
        dropout=0,
        repeat=1,
        norm_type=None,
        round=False,
        **kwargs,
    ):
        super().__init__()
        dilation_rate = 1
        in_channels = in_channels
        block = nn.ModuleList()
        for _ in range(repeat):
            inner_block = nn.ModuleList()

            inner_block.append(
                BasenjiConvBlock(
                    in_channels=in_channels,
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=int(np.round(dilation_rate)),
                    norm_type=norm_type,
                    **kwargs,
                )
            )

            inner_block.append(
                BasenjiConvBlock(
                    in_channels=filters,
                    filters=in_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    **kwargs,
                )
            )

            if norm_type is None:
                raise NotImplementedError(
                    "basenji.layers.scale is not ported to PyTorch."
                )

            block.append(Residual(nn.Sequential(*inner_block)))

            dilation_rate *= rate_mult
            if round:
                dilation_rate = np.round(dilation_rate)
        self.block = nn.Sequential(*block)
        self.out_channels = in_channels

    def forward(self, x):
        return self.block(x)

BLOCK_REGISTRY["dilated_residual"] = BasenjiDilatedResidual


class BasenjiFinal(nn.Module):
    def __init__(
        self, in_features, units, activation='linear', flatten=False, **kwargs
    ) -> None:
        super().__init__()
        block = nn.ModuleList()
        if flatten:
            block.append(Rearrange('b ... -> b (...)'))
        else:
            # in this case, same as pointwise (1x1) convolution
            block.append(Rearrange('b c l -> b l c'))
        block.append(nn.Linear(in_features=in_features, out_features=units))
        block.append(ACTIVATION_LAYER_REGISTRY[activation]())
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        return self.block(x)

BLOCK_REGISTRY["final"] = BasenjiFinal