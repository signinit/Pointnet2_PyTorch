import pytorch_lightning as pl
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader
import torch.nn.functional as F

from pointnet2.data import Custom3DClassification
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2LinearSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[0, 32, 32, 64],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 0, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def training_step(self, batch, batch_idx):
        pc, labels = batch

        logits = self.forward(pc)
        values = logits[:,0]
        loss = F.mse_loss(values, labels)
        with torch.no_grad():
            acc = (values - labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):
        pc, labels = batch

        logits = self.forward(pc)
        values = logits[:,0]
        loss = F.mse_loss(values, labels)
        acc = (values - labels).float().mean()

        return dict(val_loss=loss, val_acc=acc)

    def prepare_data(self):
        self.train_dset = Custom3DClassification(self.hparams["batch_dir"], self.hparams["batch_file"], self.hparams["num_points"], train=True)
        self.val_dset = Custom3DClassification(self.hparams["batch_dir"], self.hparams["batch_file"], self.hparams["num_points"], train=False)