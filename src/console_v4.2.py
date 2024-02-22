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

from utils import InfoHolder, generate_heatmap, superimpose, to_RGB, grad_cam, save_cam, pad_with, CVSampler, SMOTESampler
from model import init_weights, AlexNet, AlexNet_light
from data import ImageDataset, ImageDataModule
from images import make_images

EPSILON = 1e-10

# Set devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = [0,1,2,3]

# Variables to generate coordinates of CAM images
x_gaps = {49: 40, 99: 20, 149: 15, 199: 10, 249: 10, 50: 40, 100: 20, 150: 15, 200: 10, 250: 10}
x_horis = {49: -10, 99: 20, 149: 15, 199: 10, 249: 10, 50: -10, 100: 20, 150: 15, 200: 10, 250: 10}
xticks = {49: [10,20,30],
      99: [25,50,75],
      149: [25,50,75,100,125],
      199: [50,100,150],
      249: [50,100,150,200],
      50: [10,20,30],
      100: [25,50,75],
      150: [25,50,75,100,125],
      200: [50,100,150],
      250: [50,100,150,200]
     }
font = ImageFont.load_default()

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--image_type', default='RP', help='type of input image: RP / GADF / GASF / TwoChannelGAF / Combined')
parser.add_argument('--full', default=False, action='store_true', help='mode of experiments: debug / full')
parser.add_argument('--light', default=False, action='store_true', help='type of AlexNet: heavy / light')
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
parser.add_argument('--tuning', action='store_true')
parser.add_argument('--cam', default=False, action='store_true')
parser.add_argument('--force_save', default=False, action='store_true')

if __name__=="__main__":
    args = parser.parse_args()
    # Configuration
    data_root = os.path.join(root_dir,"data/")
    fname_rawdata = "carbattery_method_total.txt"
    path_rawdata = os.path.join(data_root, fname_rawdata)

    flag_light = args.light
    flag_prepare_data = args.prepare_data
    flag_train = args.train
    flag_tuning = args.tuning
    flag_extract_CAM = args.cam
    flag_force_save = args.force_save
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

    early_stopping_patience = int(max_epochs * 0.4)

    # Prepare image data before training
    if flag_prepare_data:
        for input_cycle in input_cycles:
            for target_cycle in target_cycles:
                make_images(image_type=image_type, input_cycle=input_cycle, target_cycle=target_cycle, n_resampled=n_resampled, force_save=flag_force_save)
        print("Data prepared")
        if not flag_train:
            exit(0)

    n_classes = 1 # regression
    CAM_criteria = ["below0.7", "over0.7below0.75", "over0.75below0.8", "over0.8below0.85", "over0.85"]

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
            image_path = os.path.join(data_root+image_type+"_images", path_cycle)
            if not os.path.exists(image_path):
                make_images(image_type=image_type, input_cycle=input_cycle, target_cycle=target_cycle, n_resampled=n_resampled)
            print(path_cycle)

            result_path = os.path.join(root_dir,"results", image_type+"_SMOTE"+str(n_resampled), path_cycle+"_"+str(max_epochs)+"ep")
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            cam_path = os.path.join(result_path, "CAM_images")
            if not os.path.exists(cam_path):
                os.makedirs(cam_path)
            for criterion in CAM_criteria:
                if not os.path.exists(os.path.join(cam_path, criterion)):
                    os.makedirs(os.path.join(cam_path, criterion))

            image_dataset = ImageDataset(device, image_type=image_type, img_dir=image_path, n_resampled=n_resampled, n_folds=n_folds)

            result_val_idx = []
            result_preds_val = []
            result_trues_val = []
            for fold in range(n_folds):
                print(f"Fold {fold}")
                path_fold = os.path.join(image_path, f"Fold_{fold}")
                data_module = ImageDataModule(device, batch_size=batch_size, image_dataset=image_dataset, cv_fold=fold, shuffle=True)

                if flag_light:
                    model = AlexNet_light(data_module=data_module, batch_size=batch_size, learning_rate=learning_rate, hidden_dim=hidden_dim, num_classes=n_classes)
                else:
                    model = AlexNet(data_module=data_module, batch_size=batch_size, learning_rate=learning_rate, hidden_dim=hidden_dim, num_classes=n_classes)
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
                if flag_light:
                    best_model = AlexNet_light(data_module=data_module, batch_size=batch_size, learning_rate=learning_rate, hidden_dim=hidden_dim, num_classes=n_classes)
                else:
                    best_model = AlexNet(data_module=data_module, batch_size=batch_size, learning_rate=learning_rate, hidden_dim=hidden_dim, num_classes=n_classes)
                best_model = best_model.load_from_checkpoint(cb_checkpoint.kth_best_model_path, data_module=data_module, batch_size=batch_size, learning_rate=learning_rate, hidden_dim=hidden_dim, num_classes=n_classes)
                best_model.eval()
                print(f"Best model loaded\tepoch: {cb_checkpoint.kth_best_model_path.split('epoch=')[1].split('-')[0]}, MAPE: {cb_early_stopping.best_score}")

                # Validating the optimized model for current fold
                X_val, Y_val = next(iter(data_module.val_dataloader_tosave()))
                X_val = [xx.cpu() for xx in X_val] if isinstance(X_val, list) else X_val.cpu()

                fold_trues_val = Y_val.detach().cpu().numpy()
                fold_preds_val = best_model(X_val).detach().cpu().numpy()

                fold_mape = np.round(np.mean(abs((fold_trues_val-fold_preds_val)/fold_trues_val)), 4)
                fold_mse = np.round(mean_squared_error(fold_trues_val, fold_preds_val), 4)
                fold_mae = np.round(mean_absolute_error(fold_trues_val, fold_preds_val), 4)

                result_val_idx.append(image_dataset.cv_idx_dict[fold]['test'])
                result_preds_val.append(fold_preds_val)
                result_trues_val.append(fold_trues_val)

                print(f"Fold {fold}, Best model\tMAPE: {fold_mape}\tMAE: {fold_mae}")

                # Extract CAM images
                if flag_extract_CAM:
                    best_model = best_model.to(device)

                    # container = {cr: [] for cr in CAM_criteria}
                    container = {cr: {ty: [] for ty in ['RP', 'GASF', 'GADF']} for cr in CAM_criteria}
                    xs, ys = next(iter(best_model.data_module.val_dataloader_tosave()))
                    for i, name in tqdm(enumerate(list(best_model.data_module.image_dataset.cv_idx_dict[fold]['test']))):
                        if image_type == "Combined":
                            x, y = (xs[0][i], xs[1][i]), ys[i]
                            cam_out = grad_cam(best_model, x, (best_model.feature_extraction_gaf[-3],best_model.feature_extraction_rp[-3]), image_type=image_type)
                        else:
                            x, y = xs[i], ys[i]
                            cam_out = grad_cam(best_model, x, best_model.feature_extraction[-3], image_type=image_type)
                        for k in cam_out.keys():
                            cam_img = Image.fromarray(cam_out[k])

                            # Draw coordinates
                            n_pad = 10
                            cam_img_tick = np.stack([np.pad(cam_out[k][:,:,i], n_pad, pad_with, padder=255) for i in range(3)], axis=-1)
                            im = Image.fromarray(cam_img_tick)
                            im = im.resize((300,300))
                            draw = ImageDraw.Draw(im)
                            max_size = im.size[0]
                            org_size = cam_out[k].shape[0]
                            x_gap = x_gaps[org_size]
                            x_hori = x_horis[org_size]
                            [draw.text((x_gap,max_size-x_gap),"0",fill='black',font=font)]+[draw.text((int(max_size*x/org_size)-x_hori,max_size-x_gap),str(x),fill='black',font=font) for x in xticks[org_size]]+[draw.text((max_size-x_gap-n_pad,max_size-x_gap),str(org_size),fill='black',font=font)]
                            [draw.text((x_gap-5,x_gap),"0",fill='black',font=font)]+[draw.text((x_gap-10,int(max_size*x/org_size)-x_hori),str(x),fill='black',font=font) for x in xticks[org_size]]+[draw.text((x_gap-10,max_size-x_gap-n_pad),str(org_size),fill='black',font=font)]
                            criterion = save_cam(best_model.data_module.image_dataset.train_idx[name], im, img_labels=best_model.data_module.image_dataset.img_labels, cam_path=cam_path, filename=f"[{k}]{best_model.data_module.image_dataset.train_idx[name]}")
                            if k == 'RP':
                                cam_out_ = np.pad(cam_out[k], ((0,1), (0,1), (0,0)), 'constant')
                            else:
                                cam_out_ = cam_out[k]
                            # container[criterion].append(cam_out_)
                            container[criterion][k].append(cam_out_)
                torch.cuda.empty_cache()

            # Aggregate CAM images for each criterion
            # container = {cr: np.array(container[cr]) for cr in CAM_criteria}
            container = {cr: {ty: np.array(container[cr][ty]) for ty in ['RP', 'GASF', 'GADF']} for cr in CAM_criteria}
            for cr in container.keys():
                for k in container[cr].keys():
                    if len(container[cr][k])==0: continue
                    cam_aggregated = np.round(container[cr][k].mean(axis=0)).astype(np.uint8)
                    cam_aggregated_tick = np.stack([np.pad(cam_aggregated[:,:,i], n_pad, pad_with, padder=255) for i in range(3)], axis=-1)
                    im = Image.fromarray(cam_aggregated_tick)
                    im = im.resize((300,300))
                    draw = ImageDraw.Draw(im)
                    max_size = im.size[0]
                    org_size = cam_aggregated.shape[0]
                    x_gap = x_gaps[org_size]
                    x_hori = x_horis[org_size]
                    [draw.text((x_gap,max_size-x_gap),"0",fill='black',font=font)]+[draw.text((int(max_size*x/org_size)-x_hori,max_size-x_gap),str(x),fill='black',font=font) for x in xticks[org_size]]+[draw.text((max_size-x_gap-n_pad,max_size-x_gap),str(org_size),fill='black',font=font)]
                    [draw.text((x_gap-5,x_gap),"0",fill='black',font=font)]+[draw.text((x_gap-10,int(max_size*x/org_size)-x_hori),str(x),fill='black',font=font) for x in xticks[org_size]]+[draw.text((x_gap-10,max_size-x_gap-n_pad),str(org_size),fill='black',font=font)]
                    # im.save(os.path.join(result_path, "[avg_CAM]"+path_cycle+"_"+k+".png"))
                    im.save(os.path.join(result_path, "[avg_CAM]"+k+"_"+path_cycle+"_"+cr+".png"))

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
            print(f"MAPE: {result_mape}\nMSE: {result_mse}\nR2: {result_r2}\nMAE: {result_mae}")

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
