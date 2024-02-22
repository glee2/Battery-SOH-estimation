# Set root directory
root_dir = '/home2/glee/Hyundai/MSSP/'

import os
import copy
import functools
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(root_dir)
from labels import LABELS

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from torchmetrics.functional import accuracy, recall
import torchmetrics

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, confusion_matrix, mean_absolute_error
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

from utils import LogCoshLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = [0,1,2,3]

EPSILON = 1e-10

# function for weight initialization
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.zeros_(m.bias)
        torch.nn.init.xavier_uniform_(m.weight)

# CNN-based Classification model
class AlexNet(LightningModule):
    def __init__(self, data_module=None, hidden_dim=256, batch_size=32, max_epochs=10, learning_rate=1e-3, num_classes=None):
        super(AlexNet, self).__init__()

        self.data_module = data_module
        self.image_type = self.data_module.image_type
        self.input_dim = data_module.get_input_dim()
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.loss_func = LogCoshLoss()

        self.logged_metrics = {i: [] for i in ["loss_val"]}
        self.metric_mape = torchmetrics.MeanAbsolutePercentageError()
        self.metric_mse = torchmetrics.MeanSquaredError()
        self.metric_mae = torchmetrics.MeanAbsoluteError()

        self.metric_preds = torchmetrics.CatMetric()
        self.metric_trues = torchmetrics.CatMetric()

        self.preds_val_out = np.zeros((len(self.data_module.image_dataset.cv_idx_dict[self.data_module.cv_fold]['test']))).astype(np.int64)
        self.trues_val_out = np.zeros((len(self.data_module.image_dataset.cv_idx_dict[self.data_module.cv_fold]['test']))).astype(np.int64)

        if self.image_type == "Combined":
            self.feature_extraction_gaf = self.set_layers(layer_type="feature_extraction", input_channel=self.input_dim[0][0])
            self.feature_extraction_rp = self.set_layers(layer_type="feature_extraction", input_channel=self.input_dim[1][0])
        else:
            self.feature_extraction = self.set_layers(layer_type="feature_extraction", input_channel=self.input_dim[0])
        self.regressor = self.set_layers(layer_type="regressor")
        self.avg_pool = nn.AvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def get_flattened_length(self):
        if self.image_type == "Combined":
            return functools.reduce(operator.mul, list(self.feature_extraction_gaf(torch.rand(1, *self.input_dim[0])).shape)) * 2
        else:
            return functools.reduce(operator.mul, list(self.feature_extraction(torch.rand(1, *self.input_dim)).shape))

    def set_layers(self, layer_type=None, input_channel=None):
        if layer_type == "feature_extraction":
            return nn.Sequential(
              nn.Conv2d(in_channels=input_channel,out_channels=96,kernel_size=11,stride=4,padding=2,bias=False),
              nn.ReLU(inplace=True),
              nn.BatchNorm2d(96),
              nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
              nn.Conv2d(in_channels=96,out_channels=self.hidden_dim,kernel_size=5,stride=1,padding=2,bias=False),
              nn.ReLU(inplace=True),
              nn.BatchNorm2d(self.hidden_dim),
              nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
              nn.Conv2d(in_channels=self.hidden_dim,out_channels=self.hidden_dim,kernel_size=3,stride=1,padding=2,bias=False),
              nn.ReLU(inplace=True),
              nn.BatchNorm2d(self.hidden_dim),
              nn.Conv2d(in_channels=self.hidden_dim,out_channels=self.hidden_dim,kernel_size=3,stride=1,padding=2,bias=False),
              nn.ReLU(inplace=True),
              nn.BatchNorm2d(self.hidden_dim),
              nn.Conv2d(in_channels=self.hidden_dim,out_channels=self.num_classes,kernel_size=3,stride=1,padding=2,bias=False),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            )
        elif layer_type == "regressor":
            flattened_length = self.get_flattened_length()
            return nn.Sequential(
              nn.BatchNorm1d(flattened_length),
              nn.Dropout(p=0.5),
              nn.Linear(in_features=flattened_length,out_features=4096),
              nn.ReLU(inplace=True),
              nn.Dropout(p=0.5),
              nn.Linear(in_features=4096, out_features=4096),
              nn.ReLU(inplace=True),
              nn.Linear(in_features=4096, out_features=self.num_classes),
            )
        else:
            return None

    def Flatten(self, inputs):
        shapes = inputs.size(1) * inputs.size(2) * inputs.size(3)
        outputs = inputs.view(inputs.size(0), shapes)
        return outputs

    def forward(self,x):
        if self.image_type == "Combined":
            x_gaf = self.feature_extraction_gaf(x[0])
            x_gaf = self.avg_pool(x_gaf)
            x_gaf = torch.flatten(x_gaf, 1)
            x_rp = self.feature_extraction_rp(x[1])
            x_rp = self.avg_pool(x_rp)
            x_rp = torch.flatten(x_rp, 1)
            x = torch.cat((x_gaf, x_rp), -1)
        else:
            x = self.feature_extraction(x)
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
        out = self.regressor(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze_().float()
        logits = self(x)

        metrics = self.regression_step("train", logits, y)
        return metrics["loss_train"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze_().float()
        logits = self(x)
        metrics = self.regression_step("val", logits, y)

        return metrics

    def validation_step_end(self, outputs):
        mape = self.metric_mape(outputs['preds_val'], outputs['trues_val'])
        mse = self.metric_mse(outputs['preds_val'], outputs['trues_val'])
        mae = self.metric_mae(outputs['preds_val'], outputs['trues_val'])

        trues = self.metric_trues.update(outputs['trues_val']+EPSILON)
        preds = self.metric_preds.update(outputs['preds_val']+EPSILON)

        return outputs

    def regression_step(self, level, logits, y):
        logits = logits.view(-1)
        # if logits.shape != y.shape:
        #     print(f"logit shape: {logits.shape}, y shape: {y.shape}")
        if y.size() == torch.Size([]):
            y = y.view(-1)

        loss = self.loss_func(logits, y)
        # loss = F.smooth_l1_loss(logits, y)
        # loss = F.mse_loss(logits, y)
        mape = torch.mean(abs((y-logits)/logits))
        mse = F.mse_loss(logits, y)
        mae = torch.mean(abs(y-logits))

        metrics = {
            f"loss_{level}": loss,
            f"mape_{level}": mape,
            f"mse_{level}": mse,
            f"mae_{level}": mae,
            f"trues_{level}": y,
            f"preds_{level}": logits
        }

        return metrics

    def validation_epoch_end(self, outputs):
        self.preds_val_out = copy.deepcopy(self.metric_preds.compute()-EPSILON)
        self.trues_val_out = copy.deepcopy(self.metric_trues.compute()-EPSILON)

        # Output size is different between single and multi gpu
        avg_loss = torch.cat([x["loss_val"] for x in outputs]).mean() if outputs[0]["loss_val"].size() == torch.Size([1]) else torch.cat([x["loss_val"].unsqueeze(dim=0) for x in outputs]).mean()
        avg_mape = torch.cat([x["mape_val"] for x in outputs]).mean() if outputs[0]["mape_val"].size() == torch.Size([1]) else torch.cat([x["mape_val"].unsqueeze(dim=0) for x in outputs]).mean()
        avg_mse = torch.cat([x["mse_val"] for x in outputs]).mean() if outputs[0]["mse_val"].size() == torch.Size([1]) else torch.cat([x["mse_val"].unsqueeze(dim=0) for x in outputs]).mean()
        avg_mae = torch.cat([x["mae_val"] for x in outputs]).mean() if outputs[0]["mae_val"].size() == torch.Size([1]) else torch.cat([x["mae_val"].unsqueeze(dim=0) for x in outputs]).mean()

        self.log("loss_val", avg_loss, sync_dist=True)
        self.log("mape_val", avg_mape, sync_dist=True)
        self.log("mse_val", avg_mse, sync_dist=True)
        self.log("mae_val", avg_mae, sync_dist=True)

        print("averaged metrics:", {
            "loss_val": round(avg_loss.item(),4),
            "mape_val": round(avg_mape.item(),4),
            "mse_val": round(avg_mse.item(),4),
            "mae_val": round(avg_mae.item(),4),
        })

        self.metric_preds.reset()
        self.metric_trues.reset()
        self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=5,
        )
        return [optimizer], [scheduler]

## Light version – disabling batch normalization, using average pooling
class AlexNet_light(AlexNet):
    def __init__(self, data_module=None, hidden_dim=32, batch_size=32, max_epochs=10,  learning_rate=5e-4, num_classes=None):
        super(AlexNet_light, self).__init__(data_module, hidden_dim, batch_size, max_epochs, learning_rate, num_classes)

    def set_layers(self, layer_type=None, input_channel=None):
        if layer_type == "feature_extraction":
            return nn.Sequential(
              nn.Conv2d(in_channels=input_channel,out_channels=self.hidden_dim,kernel_size=11,stride=4,padding=2,bias=False),
              nn.ReLU(inplace=True),
              # nn.BatchNorm2d(self.hidden_dim)
              nn.AvgPool2d(kernel_size=3,stride=2,padding=0),
              nn.Conv2d(in_channels=self.hidden_dim,out_channels=self.hidden_dim,kernel_size=5,stride=1,padding=2,bias=False),
              nn.ReLU(inplace=True),
              # nn.BatchNorm2d(self.hidden_dim),
              nn.AvgPool2d(kernel_size=3,stride=2,padding=0),
            )
        elif layer_type == "regressor":
            flattened_length = self.get_flattened_length()
            return nn.Sequential(
              # nn.BatchNorm1d(flattened_length),
              nn.Dropout(p=0.7),
              nn.Linear(in_features=flattened_length, out_features=self.num_classes),
            )
        else:
            return None

class RNNBase(LightningModule):
    def __init__(self, data_module=None, hidden_dim=32, num_layers=2, batch_size=32, max_epochs=10, learning_rate=5e-4):
        super(RNNBase, self).__init__()
        self.data_module = data_module
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

        self.seq_len, self.n_features = self.data_module.get_input_dim()

        self.loss_func = LogCoshLoss()

        self.logged_metrics = {i: [] for i in ["loss_val"]}
        self.metric_mape = torchmetrics.MeanAbsolutePercentageError()
        self.metric_mse = torchmetrics.MeanSquaredError()
        self.metric_mae = torchmetrics.MeanAbsoluteError()

        self.metric_preds = torchmetrics.CatMetric()
        self.metric_trues = torchmetrics.CatMetric()

        self.preds_val_out = np.zeros((len(self.data_module.ts_dataset.cv_idx_dict[self.data_module.cv_fold]['test']))).astype(np.int64)
        self.trues_val_out = np.zeros((len(self.data_module.ts_dataset.cv_idx_dict[self.data_module.cv_fold]['test']))).astype(np.int64)

        self.feature_extraction = self.set_layers()
        self.regressor = nn.Linear(hidden_dim, 1)

    def set_layers(self):
        return nn.RNN(input_size=self.n_features, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=0.5, batch_first=True)

    def forward(self, x):
        # (batch_size, seq_len, n_features)
        x, _ = self.feature_extraction(x)
        out = self.regressor(x[:,-1])

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=5,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze_().float()
        logits = self(x)

        metrics = self.regression_step("train", logits, y)
        return metrics["loss_train"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze_().float()
        logits = self(x)
        metrics = self.regression_step("val", logits, y)

        return metrics

    def validation_step_end(self, outputs):
        mape = self.metric_mape(outputs['preds_val'], outputs['trues_val'])
        mse = self.metric_mse(outputs['preds_val'], outputs['trues_val'])
        mae = self.metric_mae(outputs['preds_val'], outputs['trues_val'])

        trues = self.metric_trues.update(outputs['trues_val']+EPSILON)
        preds = self.metric_preds.update(outputs['preds_val']+EPSILON)

        return outputs

    def regression_step(self, level, logits, y):
        logits = logits.view(-1)
        if y.size() == torch.Size([]):
            y = y.view(-1)

        loss = self.loss_func(logits, y)

        mape = torch.mean(abs((y-logits)/logits))
        mse = F.mse_loss(logits, y)
        mae = torch.mean(abs(y-logits))

        metrics = {
            f"loss_{level}": loss,
            f"mape_{level}": mape,
            f"mse_{level}": mse,
            f"mae_{level}": mae,
            f"trues_{level}": y,
            f"preds_{level}": logits
        }

        return metrics

    def validation_epoch_end(self, outputs):
        self.preds_val_out = copy.deepcopy(self.metric_preds.compute()-EPSILON)
        self.trues_val_out = copy.deepcopy(self.metric_trues.compute()-EPSILON)

        # Output size is different between single and multi gpu
        avg_loss = torch.cat([x["loss_val"] for x in outputs]).mean() if outputs[0]["loss_val"].size() == torch.Size([1]) else torch.cat([x["loss_val"].unsqueeze(dim=0) for x in outputs]).mean()
        avg_mape = torch.cat([x["mape_val"] for x in outputs]).mean() if outputs[0]["mape_val"].size() == torch.Size([1]) else torch.cat([x["mape_val"].unsqueeze(dim=0) for x in outputs]).mean()
        avg_mse = torch.cat([x["mse_val"] for x in outputs]).mean() if outputs[0]["mse_val"].size() == torch.Size([1]) else torch.cat([x["mse_val"].unsqueeze(dim=0) for x in outputs]).mean()
        avg_mae = torch.cat([x["mae_val"] for x in outputs]).mean() if outputs[0]["mae_val"].size() == torch.Size([1]) else torch.cat([x["mae_val"].unsqueeze(dim=0) for x in outputs]).mean()

        self.log("loss_val", avg_loss, sync_dist=True)
        self.log("mape_val", avg_mape, sync_dist=True)
        self.log("mse_val", avg_mse, sync_dist=True)
        self.log("mae_val", avg_mae, sync_dist=True)

        print("averaged metrics:", {
            "loss_val": round(avg_loss.item(),4),
            "mape_val": round(avg_mape.item(),4),
            "mse_val": round(avg_mse.item(),4),
            "mae_val": round(avg_mae.item(),4),
        })

        self.metric_preds.reset()
        self.metric_trues.reset()
        self.train()

class LSTMRegressor(RNNBase):
    def __init__(self, data_module=None, hidden_dim=32, num_layers=2, batch_size=32, max_epochs=10, learning_rate=5e-4):
        super(LSTMRegressor, self).__init__(data_module, hidden_dim, num_layers, batch_size, max_epochs, learning_rate)

    def set_layers(self):
        return nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=0.5, batch_first=True)

class GRURegressor(RNNBase):
    def __init__(self, data_module=None, hidden_dim=32, num_layers=2, batch_size=32, max_epochs=10, learning_rate=5e-4):
        super(GRURegressor, self).__init__(data_module, hidden_dim, num_layers, batch_size, max_epochs, learning_rate)

    def set_layers(self):
        return nn.GRU(input_size=self.n_features, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=0.5, batch_first=True)

# class LSTMRegressor(LightningModule):
#     def __init__(self, data_module=None, hidden_dim=32, num_layers=2, batch_size=32, max_epochs=10, learning_rate=5e-4):
#         super(LSTMRegressor, self).__init__()
#         self.data_module = data_module
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.max_epochs = max_epochs
#         self.learning_rate = learning_rate
#
#         self.seq_len, self.n_features = self.data_module.get_input_dim()
#
#         self.loss_func = LogCoshLoss()
#
#         self.logged_metrics = {i: [] for i in ["loss_val"]}
#         self.metric_mape = torchmetrics.MeanAbsolutePercentageError()
#         self.metric_mse = torchmetrics.MeanSquaredError()
#         self.metric_mae = torchmetrics.MeanAbsoluteError()
#
#         self.metric_preds = torchmetrics.CatMetric()
#         self.metric_trues = torchmetrics.CatMetric()
#
#         self.preds_val_out = np.zeros((len(self.data_module.ts_dataset.cv_idx_dict[self.data_module.cv_fold]['test']))).astype(np.int64)
#         self.trues_val_out = np.zeros((len(self.data_module.ts_dataset.cv_idx_dict[self.data_module.cv_fold]['test']))).astype(np.int64)
#
#         self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=0.5, batch_first=True)
#         self.regressor = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         # (batch_size, seq_len, n_features)
#         x, _ = self.lstm(x)
#         out = self.regressor(x[:,-1])
#
#         return out
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             optimizer=optimizer,
#             T_0=5,
#         )
#         return [optimizer], [scheduler]
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y = y.squeeze_().float()
#         logits = self(x)
#
#         metrics = self.regression_step("train", logits, y)
#         return metrics["loss_train"]
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y = y.squeeze_().float()
#         logits = self(x)
#         metrics = self.regression_step("val", logits, y)
#
#         return metrics
#
#     def validation_step_end(self, outputs):
#         mape = self.metric_mape(outputs['preds_val'], outputs['trues_val'])
#         mse = self.metric_mse(outputs['preds_val'], outputs['trues_val'])
#         mae = self.metric_mae(outputs['preds_val'], outputs['trues_val'])
#
#         trues = self.metric_trues.update(outputs['trues_val']+EPSILON)
#         preds = self.metric_preds.update(outputs['preds_val']+EPSILON)
#
#         return outputs
#
#     def regression_step(self, level, logits, y):
#         logits = logits.view(-1)
#         if y.size() == torch.Size([]):
#             y = y.view(-1)
#
#         loss = self.loss_func(logits, y)
#
#         mape = torch.mean(abs((y-logits)/logits))
#         mse = F.mse_loss(logits, y)
#         mae = torch.mean(abs(y-logits))
#
#         metrics = {
#             f"loss_{level}": loss,
#             f"mape_{level}": mape,
#             f"mse_{level}": mse,
#             f"mae_{level}": mae,
#             f"trues_{level}": y,
#             f"preds_{level}": logits
#         }
#
#         return metrics
#
#     def validation_epoch_end(self, outputs):
#         self.preds_val_out = copy.deepcopy(self.metric_preds.compute()-EPSILON)
#         self.trues_val_out = copy.deepcopy(self.metric_trues.compute()-EPSILON)
#
#         # Output size is different between single and multi gpu
#         avg_loss = torch.cat([x["loss_val"] for x in outputs]).mean() if outputs[0]["loss_val"].size() == torch.Size([1]) else torch.cat([x["loss_val"].unsqueeze(dim=0) for x in outputs]).mean()
#         avg_mape = torch.cat([x["mape_val"] for x in outputs]).mean() if outputs[0]["mape_val"].size() == torch.Size([1]) else torch.cat([x["mape_val"].unsqueeze(dim=0) for x in outputs]).mean()
#         avg_mse = torch.cat([x["mse_val"] for x in outputs]).mean() if outputs[0]["mse_val"].size() == torch.Size([1]) else torch.cat([x["mse_val"].unsqueeze(dim=0) for x in outputs]).mean()
#         avg_mae = torch.cat([x["mae_val"] for x in outputs]).mean() if outputs[0]["mae_val"].size() == torch.Size([1]) else torch.cat([x["mae_val"].unsqueeze(dim=0) for x in outputs]).mean()
#
#         self.log("loss_val", avg_loss, sync_dist=True)
#         self.log("mape_val", avg_mape, sync_dist=True)
#         self.log("mse_val", avg_mse, sync_dist=True)
#         self.log("mae_val", avg_mae, sync_dist=True)
#
#         print("averaged metrics:", {
#             "loss_val": round(avg_loss.item(),4),
#             "mape_val": round(avg_mape.item(),4),
#             "mse_val": round(avg_mse.item(),4),
#             "mae_val": round(avg_mae.item(),4),
#         })
#
#         self.metric_preds.reset()
#         self.metric_trues.reset()
#         self.train()
