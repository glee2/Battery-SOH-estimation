# Set root directory
root_dir = '/home2/glee/Hyundai/MSSP/'

import os
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

from battery import Battery

import sys
sys.path.append(root_dir)
from labels import LABELS
from utils import SMOTESampler

from pyts.image import RecurrencePlot, GramianAngularField
from scipy.fft import fft
from six.moves import cPickle
from PIL import Image

data_root = os.path.join(root_dir,'data/')

def make_images(image_type=None, input_cycle=None, target_cycle=None, n_resampled=100, force_save=False):
    path_cycle = str(input_cycle)+"_"+str(target_cycle)
    print(path_cycle)
    image_path = os.path.join(data_root,image_type+"_images", path_cycle)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    battery_data = Battery(path=data_root, method='method_total')
    battery_data.make_inputs(input_cycle=input_cycle, target_cycle=target_cycle)

    if n_resampled > 0:
        smote_sampler = SMOTESampler(battery_data.X, battery_data.Y_label_multi, target_cycle=target_cycle, method=['SMOTE', 'BorderlineSMOTE'])
        X_res, Y_res, Y_res_labeled, oversampled_idx_dict = smote_sampler.resampling(n_resampled)
        X_res = battery_data.slicing(X_res.T, input_cycle).T
        oversampled_idx = pd.Index(np.concatenate(list(oversampled_idx_dict.values())))
        Y_res.columns = ['label']
    else:
        X_res = battery_data.X_sliced
        Y_res = battery_data.Y
        Y_res_labeled = battery_data.Y_label_multi
        oversampled_idx = pd.Index([])

    for idx in tqdm(X_res.index):
        if image_type in ["RP", "GASF", "GADF"]:
            if os.path.isfile(os.path.join(image_path, f"{image_type}_{idx}.jpg")) and not force_save:
                continue
            else:
                save_image(image_path, image_type, idx, X_res.loc[idx])
        elif image_type == "TwoChannelGAF":
            if os.path.isfile(os.path.join(image_path, f"GASF_{idx}.jpg")) and os.path.isfile(os.path.join(image_path, f"GASF_{idx}.jpg")) and not force_save:
                continue
            else:
                save_image(image_path, image_type, idx, X_res.loc[idx])
        elif image_type == "Combined":
            if os.path.isfile(os.path.join(image_path, f"GASF_{idx}.jpg")) and os.path.isfile(os.path.join(image_path, f"GADF_{idx}.jpg")) and os.path.isfile(os.path.join(image_path, f"RP_{idx}.jpg")) and not force_save:
                continue
            else:
                save_image(image_path, image_type, idx, X_res.loc[idx])

    pd.Series(battery_data.Y.index).to_csv(os.path.join(image_path, f"original_idx.csv"), index=False, header=['index'])
    pd.Series(Y_res.index).to_csv(os.path.join(image_path, f"whole_idx.csv"), index=False, header=['index'])
    pd.Series(oversampled_idx).to_csv(os.path.join(image_path, f"oversampled_idx.csv"), index=False, header=['index'])
    Y_res.to_csv(os.path.join(image_path, "annotations.csv"), index=False, header=['label'])
    Y_res_labeled.to_csv(os.path.join(image_path, "annotations_labeled.csv"), index=False, header=['label'])

def save_image(image_path, image_type, idx, sample):
    if image_type == "RP":
        rp_model = RecurrencePlot(dimension=2)
        rp_image = rp_model.fit_transform(sample.values.reshape(1, -1))
        rp_image_normalized = (rp_image - rp_image.min()) / (rp_image.max() - rp_image.min())

        image_to_save = np.round(rp_image_normalized[0] * 255).astype(np.uint8)

        im = Image.fromarray(image_to_save).convert('L')
        im.save(os.path.join(image_path, f"{image_type}_{idx}.jpg"))
    elif image_type == "GADF":
        gaf_model = GramianAngularField(method='difference')
        gaf_image = gaf_model.fit_transform(sample.values.reshape(1,-1))
        gaf_image_normalized = cv2.normalize(gaf_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        image_to_save = gaf_image_normalized[0]

        im = Image.fromarray(image_to_save).convert('L')
        im.save(os.path.join(image_path, f"{image_type}_{idx}.jpg"))
    elif image_type == "GASF":
        gaf_model = GramianAngularField(method='summation')
        gaf_image = gaf_model.fit_transform(sample.values.reshape(1,-1))
        gaf_image_normalized = cv2.normalize(gaf_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        image_to_save = gaf_image_normalized[0]

        im = Image.fromarray(image_to_save).convert('L')
        im.save(os.path.join(image_path, f"{image_type}_{idx}.jpg"))
    elif image_type == "TwoChannelGAF":
        gasf_model = GramianAngularField(method='summation')
        gasf_image = gasf_model.fit_transform(sample.values.reshape(1,-1))
        gasf_image_normalized = cv2.normalize(gasf_image, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)

        im_gasf = Image.fromarray(gasf_image_normalized[0]).convert("L")
        im_gasf.save(os.path.join(image_path, f"GASF_{idx}.jpg"))

        gadf_model = GramianAngularField(method='difference')
        gadf_image = gadf_model.fit_transform(sample.values.reshape(1,-1))
        gadf_image_normalized = cv2.normalize(gadf_image, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)

        im_gadf = Image.fromarray(gadf_image_normalized[0]).convert("L")
        im_gadf.save(os.path.join(image_path, f"GADF_{idx}.jpg"))
    elif image_type == "Combined":
        gasf_model = GramianAngularField(method='summation')
        gasf_image = gasf_model.fit_transform(sample.values.reshape(1,-1))
        gasf_image_normalized = cv2.normalize(gasf_image, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
        im_gasf = Image.fromarray(gasf_image_normalized[0]).convert("L")
        im_gasf.save(os.path.join(image_path, f"GASF_{idx}.jpg"))

        gadf_model = GramianAngularField(method='difference')
        gadf_image = gadf_model.fit_transform(sample.values.reshape(1,-1))
        gadf_image_normalized = cv2.normalize(gadf_image, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
        im_gadf = Image.fromarray(gadf_image_normalized[0]).convert("L")
        im_gadf.save(os.path.join(image_path, f"GADF_{idx}.jpg"))

        rp_model = RecurrencePlot(dimension=2)
        rp_image = rp_model.fit_transform(sample.values.reshape(1, -1))
        rp_image_normalized = (rp_image - rp_image.min()) / (rp_image.max() - rp_image.min())
        image_to_save_rp = np.round(rp_image_normalized[0] * 255).astype(np.uint8)
        im_rp = Image.fromarray(image_to_save_rp).convert('L')
        im_rp.save(os.path.join(image_path, f"RP_{idx}.jpg"))
