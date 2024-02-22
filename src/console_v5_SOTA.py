# Set root directory
root_dir = '/home2/glee/Hyundai/MSSP/'

import os
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import time
import datetime

import sys
sys.path.append(root_dir)
from labels import LABELS

from PIL import Image, ImageDraw, ImageFont

import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=FutureWarning)
warnings.filterwarnings(module='tensorflow*', action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from torchmetrics.functional import accuracy, recall
import torchmetrics
from torchvision.transforms import transforms

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, confusion_matrix, r2_score, mean_absolute_error
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

from utils import CVSampler, SMOTESampler
from model import init_weights, AlexNet, AlexNet_light, LSTMRegressor, RNNBase, GRURegressor
from data import ImageDataset, ImageDataModule, TSDataset, TSDataModule

import elm

EPSILON = 1e-10

# Set devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = [0,1,2,3]

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--image_type', default='LSTM', help='type of input image: LSTM / PF / ELM')
parser.add_argument('--full', default=False, action='store_true', help='mode of experiments: debug / full')
parser.add_argument('--max_epochs',  default=2, type=int, help='max number of epochs')
parser.add_argument('--n_folds', default=2, type=int, help='number of folds for cross validation')
parser.add_argument('--input_cycles', default=None, help='input cycles')
parser.add_argument('--target_cycles', default=None, help='target cycles')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate')
parser.add_argument('--hidden_dim', default=32, type=int, help='hidden dim')
parser.add_argument('--n_resampled', default=100, type=int, help='number of resampled data using SMOTE')
parser.add_argument('--prepare_data', action='store_true')
parser.add_argument('--train', action='store_true')

if __name__=="__main__":
    args = parser.parse_args()
    # Configuration
    data_root = os.path.join(root_dir,"data/")
    fname_rawdata = "carbattery_method_total.txt"
    path_rawdata = os.path.join(data_root, fname_rawdata)

    flag_prepare_data = args.prepare_data
    flag_train = args.train
    image_type = args.image_type

    if args.input_cycles != None:
        input_cycles = [int(i) for i in args.input_cycles.replace(']','').replace('[','').split(',')]
    else:
        input_cycles = [100]
    if args.target_cycles != None:
        target_cycles = [int(i) for i in args.target_cycles.replace(']','').replace('[','').split(',')]
    else:
        target_cycles = [700]
    max_epochs = args.max_epochs
    n_folds = args.n_folds

    if args.full:
        input_cycles = [50,100,150,200,250]
        target_cycles = [300,500,700]

    n_resampled = args.n_resampled
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    hidden_dim = args.hidden_dim

    num_layers = 2

    early_stopping_patience = int(max_epochs * 0.4)
    n_classes = 1 # regression

    if not flag_train:
        raise Exception("argument \"train\" is not passed")

    # Training
    device_id = 1
    for input_cycle in input_cycles:
        for target_cycle in target_cycles:
            start_time = time.time()
            torch.cuda.set_device(device_id % torch.cuda.device_count())
            print("Current CUDA device:",torch.cuda.current_device())
            path_cycle = str(input_cycle)+"_"+str(target_cycle)
            print(path_cycle)

            result_path = os.path.join(root_dir,"results", image_type+"_SMOTE"+str(n_resampled), path_cycle+"_"+str(max_epochs)+"ep")
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            if image_type == "ELM":
                # autoregressive
                ts_dataset = TSDataset(device, data_root=data_root, input_cycle=target_cycle, target_cycle=target_cycle, n_resampled=n_resampled, n_folds=5)
            else:
                ts_dataset = TSDataset(device, data_root=data_root, input_cycle=input_cycle, target_cycle=target_cycle, n_resampled=n_resampled, n_folds=5)

            result_val_idx = []
            result_preds_val = []
            result_trues_val = []
            for fold in range(n_folds):
                print(f"Fold {fold}")
                data_module = TSDataModule(device, ts_dataset, cv_fold=fold, batch_size=batch_size, shuffle=True)

                if image_type == "RNN":
                    model = RNNBase(data_module=data_module, hidden_dim=hidden_dim, num_layers=num_layers, batch_size=batch_size, max_epochs=max_epochs, learning_rate=learning_rate)
                elif image_type == "LSTM":
                    model = LSTMRegressor(data_module=data_module, hidden_dim=hidden_dim, num_layers=num_layers, batch_size=batch_size, max_epochs=max_epochs, learning_rate=learning_rate)
                elif image_type == "GRU":
                    model = GRURegressor(data_module=data_module, hidden_dim=hidden_dim, num_layers=num_layers, batch_size=batch_size, max_epochs=max_epochs, learning_rate=learning_rate)

                if image_type in ["RNN", "LSTM", "GRU"]:
                    model = model.to(device)

                    model_path = os.path.join(root_dir,"models", image_type+"_SMOTE"+str(n_resampled))
                    model_file_path = os.path.join(model_path, path_cycle+"_fold"+str(fold)+"_"+str(max_epochs)+"ep"+".ckpt")
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)

                    monitoring_metric = "mape_val"
                    cb_early_stopping = EarlyStopping(monitor=monitoring_metric, patience=early_stopping_patience, mode="min", verbose=True)
                    cb_checkpoint = ModelCheckpoint(monitor=monitoring_metric, save_top_k=1, mode="min", dirpath=model_path)

                    # Data parallel training
                    trainer = Trainer(gpus=[0,1,2,3] if device.type == 'cuda' else None, accelerator='dp', max_epochs=max_epochs, callbacks=[cb_checkpoint, cb_early_stopping])
                    trainer.fit(model, data_module)
                    model.eval()
                    trainer.save_checkpoint(model_file_path) # save last model

                    last_model = copy.deepcopy(model)
                    last_model.eval()

                # Load best model via checkpoint callback
                    if image_type == "RNN":
                        best_model = RNNBase(data_module=data_module, hidden_dim=hidden_dim, num_layers=num_layers, batch_size=batch_size, max_epochs=max_epochs, learning_rate=learning_rate)
                    elif image_type == "LSTM":
                        best_model = LSTMRegressor(data_module=data_module, hidden_dim=hidden_dim, num_layers=num_layers, batch_size=batch_size, max_epochs=max_epochs, learning_rate=learning_rate)
                    elif image_type == "GRU":
                        best_model = GRURegressor(data_module=data_module, hidden_dim=hidden_dim, num_layers=num_layers, batch_size=batch_size, max_epochs=max_epochs, learning_rate=learning_rate)

                    best_model = best_model.load_from_checkpoint(cb_checkpoint.kth_best_model_path, data_module=data_module, batch_size=batch_size, learning_rate=learning_rate, hidden_dim=hidden_dim, num_classes=n_classes)
                    best_model.eval()
                    print(f"Best model loaded\tepoch: {cb_checkpoint.kth_best_model_path.split('epoch=')[1].split('-')[0]}, MAPE: {cb_early_stopping.best_score}")

                    # Validating the optimized model for current fold
                    X_val, Y_val = next(iter(data_module.val_dataloader_tosave()))
                    X_val = [xx.cpu() for xx in X_val] if isinstance(X_val, list) else X_val.cpu()

                    fold_trues_val = Y_val.detach().cpu().numpy()
                    fold_preds_val = best_model(X_val).detach().cpu().numpy()

                elif image_type == "PF":
                    model = None

                elif image_type == "ELM":
                    data_module.setup()
                    X_train, Y_train = next(iter(data_module.train_dataloader_tosave()))
                    X_train = X_train.detach().cpu().numpy().squeeze(2)
                    Y_train = Y_train.detach().cpu().numpy().squeeze(1)
                    # model = elm.elm(hidden_units=hidden_dim, activation_function='relu', one_hot=False, x=X_train, y=Y_train, C=0.1, elm_type='reg')
                    # beta, tr_score, tr_time = model.fit('faster1')
                    # print(f"\t\telm training score(RMSE): {np.round(tr_score,4)}, training time(s):{np.round(float(tr_time),4)}")

                    # ELM training
                    elm_regressors = []
                    for i in range(0, target_cycle-input_cycle):
                        x_tr = X_train[:, i:i+input_cycle]
                        y_tr = X_train[:, i+input_cycle]
                        elm_regressor = elm.elm(hidden_units=hidden_dim, activation_function='relu', one_hot=False, x=x_tr, y=y_tr, C=0.1, elm_type='reg')
                        beta, tr_score, tr_time = elm_regressor.fit('faster1')
                        elm_regressors.append(elm_regressor)
                        del(elm_regressor)

                    # ELM inference
                    X_val, Y_val = next(iter(data_module.val_dataloader_tosave()))
                    X_val = X_val.detach().cpu().numpy().squeeze(2)
                    Y_val = Y_val.detach().cpu().numpy().squeeze(1)

                    x_next = np.array([]).reshape(len(X_val), 0)
                    for i in range(0, target_cycle-input_cycle):
                        if i < input_cycle:
                            x_curr = np.hstack((X_val[:, i:input_cycle], x_next))
                        elif i >= input_cycle:
                            x_curr = x_next[:, i-input_cycle:]
                        y_pred = elm_regressors[i].predict(x_curr)
                        x_next = np.hstack((x_next, y_pred[:,np.newaxis]))

                    fold_trues_val = Y_val
                    fold_preds_val = x_next[:, -1]

                fold_mape = np.round(np.mean(abs((fold_trues_val-fold_preds_val)/fold_trues_val)), 4)
                fold_mse = np.round(mean_squared_error(fold_trues_val, fold_preds_val), 4)
                fold_mae = np.round(mean_absolute_error(fold_trues_val, fold_preds_val), 4)

                result_val_idx.append(ts_dataset.cv_idx_dict[fold]['test'])
                result_preds_val.append(fold_preds_val)
                result_trues_val.append(fold_trues_val)

                print(f"Fold {fold}, Best model\tMAPE: {fold_mape}\tMAE: {fold_mae}")
                torch.cuda.empty_cache()

            result_val_idx_ = np.concatenate((result_val_idx))
            result_preds_val_ = np.concatenate((result_preds_val))[np.argsort(result_val_idx_)]
            result_trues_val_ = np.concatenate((result_trues_val))[np.argsort(result_val_idx_)]

            result_true_pred = pd.concat([pd.Series(result_val_idx_), pd.Series(result_trues_val_.squeeze()), pd.Series(result_preds_val_.squeeze())], axis=1)
            result_true_pred.columns = ['Name', 'True label', 'Predicted label']
            result_true_pred.to_csv(os.path.join(result_path, "true_pred.csv"))

            result_mape = np.round(np.mean(abs((result_trues_val_-result_preds_val_)/result_trues_val_)), 4)
            result_mse = np.round(mean_squared_error(result_trues_val_, result_preds_val_), 4)
            result_r2 = np.round(r2_score(result_trues_val_, result_preds_val_), 4)
            result_mae = np.round(mean_absolute_error(result_trues_val_, result_preds_val_), 4)
            print(f"\nMAPE: {result_mape}\nMSE: {result_mse}\nR2: {result_r2}\nMAE: {result_mae}")

            num_samples = image_dataset.__len__() if n_folds == 1 else len(result_val_idx_)

            with open(os.path.join(result_path, "Metrics.txt"), 'w') as f:
                print(f"#Samples: {str(num_samples)}\nMAPE: {result_mape}\nMSE: {result_mse}\nR2: {result_r2}\nMAE: {result_mae}", file=f)

            end_time = time.time()
            print("Elapsed time:",datetime.timedelta(seconds=end_time-start_time))

    # Performance evaluation
    out = pd.DataFrame([["","","","",""]], columns=['#Samples', 'MAPE', 'MSE', 'R2 score', 'MAE'])
    for input_cycle in input_cycles:
        for target_cycle in target_cycles:
            path_cycle = str(input_cycle)+"_"+str(target_cycle)
            result_path = os.path.join(root_dir,"results", image_type+"_SMOTE"+str(n_resampled), path_cycle+"_"+str(max_epochs)+"ep")
            metric_path = os.path.join(result_path, "Metrics.txt")
            lines = []
            with open(metric_path, 'r') as f:
                for line in f:
                    lines.append(str(np.round(float(line.replace(' ','').replace('\n','').split(':')[1]), 4)))
                out = pd.concat([out, pd.DataFrame([lines], columns=['#Samples', 'MAPE', 'MSE', 'R2 score', 'MAE'], index=[path_cycle])])
    out = out.iloc[1:]
    out.to_csv(os.path.join(root_dir,"results/", image_type+"_SMOTE"+str(n_resampled), "Total_result_"+str(max_epochs)+"ep.csv"))

    torch.cuda.empty_cache()
