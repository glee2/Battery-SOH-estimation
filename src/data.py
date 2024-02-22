# Set root directory
root_dir = '/home2/glee/Hyundai/MSSP/'

import os
import copy
import pandas as pd
import numpy as np

import sys
sys.path.append(root_dir)
from labels import LABELS

from PIL import Image

from battery import Battery
from utils import CVSampler, SMOTESampler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from torchmetrics.functional import accuracy, recall
import torchmetrics
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, StratifiedShuffleSplit

from pytorch_lightning import LightningDataModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PyTorch Dataset for RP images
class ImageDataset(Dataset):
    def __init__(self, device, image_type, img_dir, n_resampled, n_folds, cross_validation=True, transform_funcs=None):
        self.device = device
        self.img_dir = img_dir
        self.image_type = image_type
        self.transform_funcs = transform_funcs
        self.n_resampled = n_resampled
        self.whole_idx = pd.read_csv(os.path.join(img_dir, "whole_idx.csv"))['index'].to_list()
        self.original_idx = pd.read_csv(os.path.join(img_dir, "original_idx.csv"))['index'].to_list()
        self.oversampled_idx_whole = pd.read_csv(os.path.join(img_dir, "oversampled_idx.csv"))['index'].to_list()

        if n_resampled > 0:
            self.oversampled_idx = self.get_oversampled_idx()
        else:
            self.oversampled_idx = []

        self.train_idx = self.original_idx + self.oversampled_idx
        self.img_labels = pd.read_csv(os.path.join(img_dir, "annotations.csv")).set_index(pd.Index(self.whole_idx)).loc[self.train_idx]
        self.img_labels_labeled = pd.read_csv(os.path.join(img_dir, "annotations_labeled.csv")).set_index(pd.Index(self.whole_idx)).loc[self.train_idx] # labeled for oversampling

        # Cross validation sampler
        self.cv_sampler = CVSampler(img_labels=self.img_labels_labeled, n_folds=n_folds)
        self.cv_idx_dict = self.cv_sampler.get_idx_dict()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, n_idx):
        idx = self.train_idx[n_idx]
        if self.image_type == "TwoChannelGAF":
            img_path_s = os.path.join(self.img_dir, f"GASF_{idx}.jpg")
            image_s = self.read_image(img_path_s)
            image_s = np.array(image_s)[np.newaxis, :, :] / 255
            image_s = torch.cuda.FloatTensor(image_s, device=device)
            img_path_d = os.path.join(self.img_dir, f"GADF_{idx}.jpg")
            image_d = self.read_image(img_path_d)
            image_d = np.array(image_d)[np.newaxis, :, :] / 255
            image_d = torch.cuda.FloatTensor(image_d, device=device)
            image = torch.cat((image_s, image_d), 0)
        elif self.image_type == "Combined":
            img_path_s = os.path.join(self.img_dir, f"GASF_{idx}.jpg")
            image_s = self.read_image(img_path_s)
            image_s = np.array(image_s)[np.newaxis, :, :] / 255
            image_s = torch.cuda.FloatTensor(image_s, device=device)

            img_path_d = os.path.join(self.img_dir, f"GADF_{idx}.jpg")
            image_d = self.read_image(img_path_d)
            image_d = np.array(image_d)[np.newaxis, :, :] / 255
            image_d = torch.cuda.FloatTensor(image_d, device=device)

            img_path_rp = os.path.join(self.img_dir, f"RP_{idx}.jpg")
            image_rp = self.read_image(img_path_rp)
            image_rp = np.array(image_rp)[np.newaxis, :, :] / 255
            image_rp = torch.cuda.FloatTensor(image_rp, device=device)

            image = (torch.cat((image_s, image_d), 0), image_rp)
        else:
            img_path = os.path.join(self.img_dir, f"{self.image_type}_{idx}.jpg")
            image = self.read_image(img_path)
            if self.transform_funcs:
                image = self.transform_funcs(image)
                image = image.to(device)
            image = np.array(image)[np.newaxis, :, :] / 255
            image = torch.cuda.FloatTensor(image, device=device)
        label = self.img_labels.loc[idx].values
        # label = torch.cuda.FloatTensor(label, device=device)
        label = torch.tensor(label, device=device, dtype=torch.float)

        return image, label

    def read_image(self, path):
        return Image.open(path).convert("L")

    def get_oversampled_idx(self):
        whole_labels = pd.read_csv(os.path.join(self.img_dir, "annotations_labeled.csv")).set_index(pd.Index(self.whole_idx))
        oversampled_labels_whole = whole_labels.copy(deep=True).loc[self.oversampled_idx_whole]
        n_labels = len(np.unique(oversampled_labels_whole))
        n_resampled_per_label = self.n_resampled - np.unique(whole_labels.loc[self.original_idx], return_counts=True)[1]
        # When n_resampled < original samples for a label
        n_resampled_per_label[n_resampled_per_label<0] = 0

        oversampled_idx = []
        for i in range(n_labels):
            oversampled_idx.append(np.random.choice(oversampled_labels_whole[oversampled_labels_whole['label']==i].index, n_resampled_per_label[i], replace=False))
        return list(np.concatenate(oversampled_idx))


# PyTorch Lightning DataModule for RP images
class ImageDataModule(LightningDataModule):
    def __init__(self, device, batch_size, image_dataset, cv_fold, transform_funcs=None, shuffle=False):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.image_dataset = image_dataset
        self.image_type = self.image_dataset.image_type
        self.img_labels = self.image_dataset.img_labels
        self.cv_fold = cv_fold
        self.shuffle = shuffle

        self.train_data = self.val_data = self.test_data = None

    def get_input_dim(self):
        temp_x, temp_y = self.image_dataset.__getitem__(0)
        if self.image_type == "Combined":
            return (temp_x[0].shape, temp_x[1].shape)
        else:
            return temp_x.shape

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if (stage == "fit" or stage is None) and self.train_data is None:
            self.train_data = Subset(self.image_dataset, self.image_dataset.cv_idx_dict[self.cv_fold]['train'])
            self.val_data = Subset(self.image_dataset, self.image_dataset.cv_idx_dict[self.cv_fold]['test'])

        if (stage == "test" or stage is None) and self.test_data is None:
            pass

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)

    def val_dataloader_tosave(self):
        return DataLoader(self.val_data, batch_size=self.val_data.__len__(), shuffle=False, drop_last=False, num_workers=0)

    def test_dataloader(self):
        return None

    def whole_dataloader(self):
        return DataLoader(self.image_dataset, batch_size=self.image_dataset.__len__(), shuffle=False, drop_last=False, num_workers=0)


# PyTorch Dataset for Time-series data
class TSDataset(Dataset):
    def __init__(self, device, data_root, input_cycle, target_cycle, n_resampled, n_folds, cross_validation=True):
        self.device = device
        self.data_root = data_root
        self.input_cycle = input_cycle
        self.target_cycle = target_cycle
        self.n_resampled = n_resampled
        self.ts_data = self.ts_labels = self.ts_labels_labeled = self.whole_idx = self.original_idx = self.oversampled_idx = None

        self.oversample()

        self.train_idx = self.original_idx + self.oversampled_idx

        # Cross validation sampler
        self.cv_sampler = CVSampler(img_labels=self.ts_labels_labeled, n_folds=n_folds)
        self.cv_idx_dict = self.cv_sampler.get_idx_dict()

    def __len__(self):
        return len(self.ts_labels)

    def __getitem__(self, n_idx):
        idx = self.train_idx[n_idx]
        ts_sample = torch.tensor(self.ts_data.loc[idx].values, device=device, dtype=torch.float)
        ts_sample = ts_sample.unsqueeze(1)
        label = torch.tensor(self.ts_labels.loc[idx].values, device=device, dtype=torch.float)

        return ts_sample, label

    def oversample(self):
        battery_data = Battery(path=self.data_root, method='method_total')
        battery_data.make_inputs(input_cycle=self.input_cycle, target_cycle=self.target_cycle)

        if self.n_resampled > 0:
            smote_sampler = SMOTESampler(battery_data.X, battery_data.Y_label_multi, target_cycle=self.target_cycle, method=['SMOTE', 'BorderlineSMOTE'])
            X_res, Y_res, Y_res_labeled, oversampled_idx_dict = smote_sampler.resampling(self.n_resampled)
            X_res = battery_data.slicing(X_res.T, self.input_cycle).T
            oversampled_idx = pd.Index(np.concatenate(list(oversampled_idx_dict.values())))
            Y_res.columns = ['label']
        else:
            X_res = battery_data.X_sliced
            Y_res = battery_data.Y
            Y_res_labeled = battery_data.Y_label_multi
            oversampled_idx = pd.Index([])

        self.ts_data = X_res
        self.ts_labels = Y_res
        self.ts_labels_labeled = Y_res_labeled
        self.whole_idx = list(Y_res.index)
        self.original_idx = list(battery_data.Y.index)
        self.oversampled_idx = list(oversampled_idx)

# PyTorch Lightning DataModule for Time-series data
class TSDataModule(LightningDataModule):
    def __init__(self, device, ts_dataset, cv_fold, batch_size, transform_funcs=None, shuffle=False):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.ts_dataset = ts_dataset
        self.ts_labels = self.ts_dataset.ts_labels
        self.cv_fold = cv_fold
        self.shuffle = shuffle

        self.train_data = self.val_data = self.test_data = None

    def get_input_dim(self):
        temp_x, temp_y = self.ts_dataset.__getitem__(0)

        return temp_x.shape

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if (stage == "fit" or stage is None) and self.train_data is None:
            self.train_data = Subset(self.ts_dataset, self.ts_dataset.cv_idx_dict[self.cv_fold]['train'])
            self.val_data = Subset(self.ts_dataset, self.ts_dataset.cv_idx_dict[self.cv_fold]['test'])

        if (stage == "test" or stage is None) and self.test_data is None:
            pass

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True, num_workers=0)

    def train_dataloader_tosave(self):
        return DataLoader(self.train_data, batch_size=self.train_data.__len__(), shuffle=True, drop_last=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)

    def val_dataloader_tosave(self):
        return DataLoader(self.val_data, batch_size=self.val_data.__len__(), shuffle=False, drop_last=False, num_workers=0)

    def test_dataloader(self):
        return None

    def whole_dataloader(self):
        return DataLoader(self.ts_dataset, batch_size=self.ts_dataset.__len__(), shuffle=False, drop_last=False, num_workers=0)
