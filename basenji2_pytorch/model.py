"""Re-implementation of Basenji2, [Cross-species regulatory sequence activity prediction](https://doi.org/10.1371/journal.pcbi.1008050).

This is a minimal re-implementation that is not compatible with the full range of configuration
supported by the Basenji GitHub to instantiate a `basenji.seqnn.SeqNN` object.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from textwrap import dedent
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchinfo
from natsort import natsorted
from pooch import retrieve
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torchmetrics import Metric, MetricCollection, PearsonCorrCoef

from .blocks import ACTIVATION_LAYER_REGISTRY, BLOCK_REGISTRY


def basenji2_weights() -> str:
    retrieve(
        url="https://zenodo.org/records/14969593/files/basenji2.pth?download=1",
        known_hash="md5:1f46acc933e05515faf4d7eec0e27d42",
    )


basenji2_params = {
    "train": {
        "batch_size": 4,
        "optimizer": "sgd",
        "learning_rate": 0.15,
        "momentum": 0.99,
        "patience": 16,
        "clip_norm": 2,
    },
    "model": {
        "seq_length": 131072,
        "target_length": 1024,
        "activation": "gelu",
        "norm_type": "batch",
        "bn_momentum": 0.9,
        "trunk": [
            {"name": "conv_block", "filters": 288, "kernel_size": 15, "pool_size": 2},
            {
                "name": "conv_tower",
                "filters_init": 339,
                "filters_mult": 1.1776,
                "kernel_size": 5,
                "pool_size": 2,
                "repeat": 6,
            },
            {
                "name": "dilated_residual",
                "filters": 384,
                "rate_mult": 1.5,
                "repeat": 11,
                "dropout": 0.3,
                "round": True,
            },
            {"name": "Cropping1D", "cropping": 64},
            {"name": "conv_block", "filters": 1536, "dropout": 0.05},
        ],
        "head_human": {"name": "final", "units": 5313, "activation": "softplus"},
    },
}


# 30,123,584 parameters
# 30,099,228 trainable parameters
class Basenji2(nn.Module):
    def __init__(self, params: Dict) -> None:
        super().__init__()
        for k, v in params.items():
            self.__setattr__(k, v)
        self.build_model()

    def build_block(self, block_params: Dict[str, Any]):
        block_args = {}
        block_name = block_params["name"]

        # set global defaults
        global_vars = ["activation", "bn_momentum", "norm_type", "padding"]
        for gv in global_vars:
            gv_value = getattr(self, gv, False)
            if gv_value:
                block_args[gv] = gv_value

        # set remaining params
        block_args.update(block_params)
        del block_args["name"]

        block = BLOCK_REGISTRY[block_name]
        return block(**block_args)

    def build_model(self):
        model_trunk = nn.ModuleList()

        CONV_LAYERS = frozenset(["conv_block", "conv_tower", "dilated_residual"])

        # trunk
        in_channels = 4
        for block_params in self.trunk:
            if block_params["name"] in CONV_LAYERS:
                block_params["in_channels"] = in_channels

            block = self.build_block(block_params)
            model_trunk.append(block)

            if block_params["name"] in CONV_LAYERS:
                in_channels = block.out_channels

        # final trunk activation
        model_trunk.append(ACTIVATION_LAYER_REGISTRY[self.activation]())
        trunk = nn.Sequential(*model_trunk)

        # head
        trunk_stats = torchinfo.summary(
            trunk, input_size=(1, 4, self.seq_length), verbose=0
        )
        trunk_out_features = trunk_stats.summary_list[-1].output_size[-2]

        head_keys = natsorted([v for v in vars(self) if v.startswith("head")])
        self.heads = [getattr(self, hk) for hk in head_keys]
        _heads = nn.ModuleDict()
        for head in self.heads:
            if not isinstance(head, list):
                head = [head]

            # build blocks
            for block_params in head:
                block_params["in_features"] = trunk_out_features
                _heads[block_params["name"]] = self.build_block(block_params)

        head = nn.Sequential(OrderedDict(_heads))

        # finish
        self.model = nn.Sequential(OrderedDict([("trunk", trunk), ("head", head)]))

    def forward(self, x, return_only_embeddings=False):
        if return_only_embeddings:
            return self.model.trunk(x)
        else:
            return self.model(x)


class PLBasenji2(pl.LightningModule):
    def __init__(
        self,
        params: Optional[Dict] = None,
        pretrained=False,
        head=True,
        metrics: Optional[List[Metric]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if params is None:
            params = basenji2_params
        model_params = params["model"]

        if not head and params is not None and not pretrained:
            warnings.warn(
                "Got head=False and non-null train parameters. Did you mean to train a headless model?"
            )

        if not head:
            model_params.pop("head_human", None)

        if params is None and not pretrained:
            self.hparams["train_params"] = params["train"]

        if not pretrained:
            self.hparams["lr"] = self.hparams.train_params["learning_rate"]  # type: ignore
            self.hparams["batch_size"] = self.hparams.train_params["batch_size"]  # type: ignore
            msg = dedent(
                """
                Use PLBasenji2.get_cross2020_trainer(...) to get a trainer matching the Basenji2 paper.
                Also use a batch size of 4 with your dataloader/datamodule, or adjust the learning
                rate proportionally. Original parameters: lr=0.15, batch_size=4
                """
            ).strip()
            print(msg)

        self.basenji2 = Basenji2(model_params)
        self.loss_fn = nn.PoissonNLLLoss(log_input=False)

        if pretrained:
            self.basenji2.load_state_dict(torch.load(basenji2_weights()), strict=False)
        else:

            @torch.no_grad()
            def init_weights(m):
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        m.weight, nonlinearity="relu"
                    )  # matches Keras, gain of sqrt(2) regardless of activation function
                    if getattr(m, "bias", False):
                        m.bias.fill_(0)

            self.basenji2.apply(init_weights)

        # TODO: allow different input sizes (not needed atm)
        model_stat = torchinfo.summary(self.basenji2, (1, 4, 131072), verbose=0)
        out_shape = model_stat.summary_list[-1].output_size

        if metrics is None:
            metrics = []
        metrics.append(PearsonCorrCoef(num_outputs=np.prod(out_shape)))
        _metrics = MetricCollection(metrics)
        self.train_metrics = _metrics.clone(prefix="train/")
        self.val_metrics = _metrics.clone(prefix="val/")
        self.test_metrics = _metrics.clone(prefix="test/")

    def forward(self, x):
        return self.basenji2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.basenji2(x)
        loss = self.loss_fn(x, y)
        self.log("train/loss", loss)
        self.log_dict(self.train_metrics(x, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.basenji2(x)
        loss = self.loss_fn(x, y)
        self.log("val/loss", loss)
        self.log_dict(self.val_metrics(x, y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.basenji2(x)
        self.log_dict(self.test_metrics(x, y))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        return self.basenji2(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.hparams.train_params["optimizer"] == "sgd":  # type: ignore
            optim = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,  # type: ignore
                momentum=self.hparams.train_params["momentum"],  # type: ignore
            )
        else:
            raise NotImplementedError("Optimizer isn't SGD")
        return optim

    @classmethod
    def get_cross2020_trainer(cls, checkpoint_dir, **kwargs) -> pl.Trainer:
        callbacks = [
            EarlyStopping("val/PearsonCorrCoef", mode="max", patience=16),
            ModelCheckpoint(
                dirpath=checkpoint_dir, filename="{epoch}-{val_PearsonCorrCoef:.2f}"
            ),
        ]
        return pl.Trainer(
            callbacks=callbacks,
            gradient_clip_val=2,
            gradient_clip_algorithm="norm",
            **kwargs,
        )
