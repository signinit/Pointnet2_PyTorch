import pytorch_lightning as pl
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader
import torch.nn.functional as F

from pointnet2.data import Custom3DLinear
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2LinearSSG(PointNet2ClassificationSSG):

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[0, 64, 64, 128],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=self.hparams["model.use_xyz"]
            )
        )

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
        print(pc.size())
        print(labels)

        logits = self.forward(pc)
        print(logits.size())
        value = logits[0]
        loss = F.mse_loss(value, labels)
        with torch.no_grad():
            acc = (value - labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):
        pc, labels = batch

        logits = self.forward(pc)
        print(logits.size())
        value = logits[0]
        loss = F.mse_loss(value, labels)
        acc = (value - labels).float().mean()

        return dict(val_loss=loss, val_acc=acc)

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

    def prepare_data(self):
        self.train_dset = Custom3DLinear(self.hparams["batch_dir"], self.hparams["batch_file"], self.hparams["num_points"], train=True)
        self.val_dset = Custom3DLinear(self.hparams["batch_dir"], self.hparams["batch_file"], self.hparams["num_points"], train=False)
