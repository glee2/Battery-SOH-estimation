# Set root directory
root_dir = '/home2/glee/Hyundai/MSSP/'

import os
import csv
import math
import copy
import functools
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import cv2

import torch

import sys
sys.path.append(root_dir)
from labels import LABELS

from pyts.image import RecurrencePlot, GramianAngularField
from PIL import Image

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, StratifiedShuffleSplit

from imblearn.over_sampling import SMOTE, BorderlineSMOTE

import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=FutureWarning)
warnings.filterwarnings(module='tensorflow*', action='ignore', category=FutureWarning)
warnings.filterwarnings(module='imblearn*', action='ignore', category=UserWarning)

# MAPE approximation
def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

# Cross validation sampler
class CVSampler:
    def __init__(self, img_labels, test_ratio=0.2, n_folds=5, random_state=10):
        self.img_labels = img_labels
        self.test_ratio = test_ratio
        self.n_folds = n_folds
        self.random_state = random_state

        self.idx_dict = {}
        self.split()

    def split(self):
        if self.n_folds == 1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.random_state)
            for train_idx, test_idx in sss.split(np.zeros(len(self.img_labels)), self.img_labels):
                self.idx_dict[0] = {'train': train_idx, 'test': test_idx}
        else:
            skf = StratifiedKFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True)
            for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(self.img_labels)), self.img_labels)):
                self.idx_dict[fold] = {'train': train_idx, 'test': test_idx}

    def get_idx_dict(self):
        return self.idx_dict

# Oversampling sampler
class SMOTESampler:
    def __init__(self, X, Y, target_cycle, method=['SMOTE', 'BorderlineSMOTE']):
        self.X = X
        self.Y = Y
        self.target_cycle = target_cycle
        self.method = method

        self.n_labels = len(np.unique(self.Y))
        self.n_samples_per_label = np.unique(self.Y, return_counts=True)[1]

    def resampling(self, n_resampled):
        n_resampled_per_method = []
        for m in range(len(self.method)):
            if m < len(self.method)-1:
                n_resampled_per_method.append(int(n_resampled/len(self.method)))
            else:
                n_resampled_per_method.append(n_resampled-int(n_resampled/len(self.method))*m)

        X_over, Y_over, Y_over_labeled = {}, {}, {}
        oversampled_idx = {}
        for m in range(len(self.method)):
            sampling_ratio = {}
            for i in range(self.n_labels):
                n_samples = self.n_samples_per_label[i]
                if n_samples > 1:
                    if n_samples < n_resampled_per_method[m]:
                        if m < len(self.method) - 1:
                            sampling_ratio[i] = n_resampled_per_method[m] + int(n_samples/len(self.method))
                        else:
                            sampling_ratio[i] = n_resampled_per_method[m] + (n_samples-int(n_samples/len(self.method))*m)
                    elif n_samples < n_resampled:
                        if m < len(self.method) - 1:
                            sampling_ratio[i] = int((n_resampled-n_samples)/len(self.method)) + n_samples
                        else:
                            sampling_ratio[i] = (n_resampled-n_samples)-int((n_resampled-n_samples)/len(self.method))*m + n_samples
                    else:
                        sampling_ratio[i] = n_samples

            k_neighbors = np.min(self.n_samples_per_label) - 1
            if self.method[m] == "SMOTE":
                sampler = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k_neighbors)
            elif self.method[m] == "BorderlineSMOTE":
                sampler = BorderlineSMOTE(sampling_strategy=sampling_ratio, k_neighbors=k_neighbors, m_neighbors=k_neighbors*2)

            X_res, Y_res = sampler.fit_resample(self.X, self.Y)

            oversampled_idx[self.method[m]] = pd.Index([self.method[m]+"_oversampled_{}".format(x) for x in range(len(X_res)-len(self.X))])

            X_over[self.method[m]] = X_res.iloc[len(self.X):].set_index(oversampled_idx[self.method[m]])
            Y_over[self.method[m]] = pd.DataFrame(X_over[self.method[m]].T.loc[self.target_cycle])
            Y_over_labeled[self.method[m]] = pd.DataFrame(Y_res.iloc[len(self.Y):]).set_index(oversampled_idx[self.method[m]])

        X_out = self.X.copy(deep=True)
        Y_out = pd.DataFrame(X_out.T.loc[self.target_cycle])
        Y_out_labeled = pd.DataFrame(self.Y.copy(deep=True))
        for m in range(len(self.method)):
            X_out = pd.concat([X_out, X_over[self.method[m]]], axis=0).set_index(X_out.index.append(oversampled_idx[self.method[m]]))
            Y_out_labeled = pd.concat([Y_out_labeled, Y_over_labeled[self.method[m]]], axis=0).set_index(X_out.index)
            Y_out = pd.concat([Y_out, Y_over[self.method[m]]], axis=0).set_index(X_out.index)

        return X_out, Y_out, Y_out_labeled, oversampled_idx

## Set of functions for generating CAM images
class InfoHolder():
    def __init__(self, heatmap_layer):
        self.gradient = None
        self.activation = None
        self.heatmap_layer = heatmap_layer

    def get_gradient(self, grad):
        self.gradient = grad

    def hook(self, model, input, output):
        output.register_hook(self.get_gradient)
        self.activation = output.detach()

def generate_heatmap(weighted_activation):
    raw_heatmap = torch.mean(weighted_activation, 0)
    heatmap = np.maximum(raw_heatmap.detach().cpu(), 0)
    heatmap /= torch.max(heatmap) + 1e-10
    return heatmap.numpy()

def superimpose(input_img, heatmap):
    img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.6 + img * 0.4)
    pil_img = cv2.cvtColor(superimposed_img,cv2.COLOR_BGR2RGB)
    return pil_img

def to_RGB(tensor):
    tensor = (tensor - tensor.min())
    tensor = tensor/(tensor.max() + 1e-10)
    image_binary = np.transpose(tensor.cpu().detach().numpy(), (1, 2, 0))
    image = np.uint8(255 * image_binary)
    return image

def grad_cam(model, input_tensor, heatmap_layer, image_type, truelabel=None):
    if image_type == "Combined":
        info_gaf = InfoHolder(heatmap_layer[0])
        heatmap_layer[0].register_forward_hook(info_gaf.hook)
        info_rp = InfoHolder(heatmap_layer[1])
        heatmap_layer[1].register_forward_hook(info_rp.hook)

        output = model((input_tensor[0].unsqueeze(0), input_tensor[1].unsqueeze(0)))
        truelabel = truelabel if truelabel else torch.argmax(output)
        output[truelabel].backward()

        weights_gaf = torch.mean(info_gaf.gradient, [0, 2, 3])
        activation_gaf = info_gaf.activation.squeeze(0)
        weighted_activation_gaf = torch.zeros(activation_gaf.shape)
        for idx, (weight_gaf, activation_gaf) in enumerate(zip(weights_gaf, activation_gaf)):
            weighted_activation_gaf[idx] = weight_gaf * activation_gaf
        weights_rp = torch.mean(info_rp.gradient, [0, 2, 3])
        activation_rp = info_rp.activation.squeeze(0)
        weighted_activation_rp = torch.zeros(activation_rp.shape)
        for idx, (weight_rp, activation_rp) in enumerate(zip(weights_rp, activation_rp)):
            weighted_activation_rp[idx] = weight_rp * activation_rp

        heatmap_gaf = generate_heatmap(weighted_activation_gaf)
        heatmap_rp = generate_heatmap(weighted_activation_rp)
        input_image_s = to_RGB(input_tensor[0][0].unsqueeze(0))
        input_image_d = to_RGB(input_tensor[0][1].unsqueeze(0))
        input_image_rp = to_RGB(input_tensor[1])
        out = {'GASF': superimpose(input_image_s, heatmap_gaf), 'GADF': superimpose(input_image_d, heatmap_gaf), 'RP': superimpose(input_image_rp, heatmap_rp)}
    else:
        info = InfoHolder(heatmap_layer)
        heatmap_layer.register_forward_hook(info.hook)
        output = model(input_tensor.unsqueeze(0))[0]
        truelabel = truelabel if truelabel else torch.argmax(output)
        output[truelabel].backward()

        weights = torch.mean(info.gradient, [0, 2, 3])
        activation = info.activation.squeeze(0)
        weighted_activation = torch.zeros(activation.shape)
        for idx, (weight, activation) in enumerate(zip(weights, activation)):
            weighted_activation[idx] = weight * activation

        heatmap = generate_heatmap(weighted_activation)
        if image_type == 'TwoChannelGAF':
            input_image_s = to_RGB(input_tensor[0].unsqueeze(0))
            input_image_d = to_RGB(input_tensor[1].unsqueeze(0))
            out = {'GASF': superimpose(input_image_s, heatmap), 'GADF': superimpose(input_image_d, heatmap)}
        else:
            input_image = to_RGB(input_tensor)
            out = {image_type: superimpose(input_image, heatmap)}

    return out

def save_cam(name, cam_img, img_labels, cam_path=None, filename=None):
    if filename == None:
        filename = name
    if name in list(img_labels[(img_labels<0.7)['label']].index):
        cam_img.save(os.path.join(cam_path, "below0.7", filename+".png"))
        criterion = "below0.7"
    elif name in list(img_labels[((img_labels>=0.7) & (img_labels<0.75))['label']].index):
        cam_img.save(os.path.join(cam_path, "over0.7below0.75", filename+".png"))
        criterion = "over0.7below0.75"
    elif name in list(img_labels[((img_labels>=0.75) & (img_labels<0.8))['label']].index):
        cam_img.save(os.path.join(cam_path, "over0.75below0.8", filename+".png"))
        criterion = "over0.75below0.8"
    elif name in list(img_labels[((img_labels>=0.8) & (img_labels<0.85))['label']].index):
        cam_img.save(os.path.join(cam_path, "over0.8below0.85", filename+".png"))
        criterion = "over0.8below0.85"
    elif name in list(img_labels[(img_labels>=0.85)['label']].index):
        cam_img.save(os.path.join(cam_path, "over0.85", filename+".png"))
        criterion = "over0.85"
    return criterion

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
