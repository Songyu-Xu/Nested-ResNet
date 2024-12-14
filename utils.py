import torch
import collections
import os
from torch.utils.data import DataLoader, ConcatDataset
from models_resnet import mono_res, mono_vit, mono_vit_mlp, mono_res_mlp, \
    stereo_res, stereo_vit, stereo_vit_mlp, stereo_res_mlp, stereo_mask_res_mlp, stereo_img_mask_res_mlp, \
    stereo_img_contour_res_mlp, stereo_img_contour_pt_res_mlp, stereo_mask_contour_res_mlp, stereo_mask_contour_pt_res_mlp, stereo_rgbm_res_mlp, \
    stereo_rgbm_mask_res_mlp, stereo_rgbm_SpaAtt_res_mlp, stereo_rgbm_mask_SpaAtt_res_mlp, stereo_res_lstm, stereo_vit_lstm,\
    stereo_resnet50_mlp, stereo_resnet18_mlp, stereo_resnet50_mask_axis, stereo_mask_disparity_res_mlp, stereo_disparity_res_mlp, \
    stereo_depth_axis_res_mlp, stereo_resnet18, stereo_vit_axis
from models_resunet import stereo_res_unet, stereo_res_unet_upconv, stereo_res_unet_upconv_img
from models_skresnet import stereo_skresnet, stereo_skresnet_depth, stereo_skresnet_axis, stereo_skresnet_axis_depth
from senseiloader import SENSEIDataset
import numpy as np


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def get_model(model, pretrained=False):
    '''
    Mono setting
    '''
    if model == 'mono_res':
        out_model = mono_res(3)
    elif model == 'mono_vit':
        out_model = mono_vit(3)
    elif model == 'mono_vit_mlp':
        out_model = mono_vit_mlp(3)
    elif model == 'mono_res_mlp':
        out_model = mono_res_mlp(3)
        '''
        Stereo setting
        '''
    elif model == 'stereo_res':
        out_model = stereo_res(6)
    elif model == 'stereo_vit':
        out_model = stereo_vit(6)
    elif model == 'stereo_vit_mlp':
        out_model = stereo_vit_mlp(6)
    elif model == 'stereo_res_mlp':
        out_model = stereo_res_mlp(6)
    elif model == 'stereo_res':
        out_model = stereo_res(6)
    elif model == 'stereo_mask_res_mlp':
        out_model = stereo_mask_res_mlp(6)
    elif model == 'stereo_img_mask_res_mlp':
        out_model = stereo_img_mask_res_mlp(6)
    elif model == 'stereo_img_contour_res_mlp':
        out_model = stereo_img_contour_res_mlp(6)
    elif model == 'stereo_img_contour_pt_res_mlp':
        out_model = stereo_img_contour_pt_res_mlp(6)
    elif model == 'stereo_mask_contour_res_mlp':
        out_model = stereo_mask_contour_res_mlp(6)
    elif model == 'stereo_mask_contour_pt_res_mlp':
        out_model = stereo_mask_contour_pt_res_mlp(6)
    elif model == 'stereo_rgbm_res_mlp':
        out_model = stereo_rgbm_res_mlp(8)
    elif model == 'stereo_rgbm_SpaAtt_res_mlp':
        out_model = stereo_rgbm_SpaAtt_res_mlp(8)
    elif model == 'stereo_rgbm_mask_res_mlp':
        out_model = stereo_rgbm_mask_res_mlp(8)
    elif model == 'stereo_rgbm_mask_SpaAtt_res_mlp':
        out_model = stereo_rgbm_mask_SpaAtt_res_mlp(8)
    elif model == 'stereo_res_lstm':
        out_model = stereo_res_lstm(6)
    elif model == 'stereo_vit_lstm':
        out_model = stereo_vit_lstm(6)
    elif model == 'stereo_res_unet':
        out_model = stereo_res_unet(6)
    elif model == 'stereo_res_unet_upconv':
        out_model = stereo_res_unet_upconv(6)
    elif model == 'stereo_res_unet_upconv_img':
        out_model = stereo_res_unet_upconv_img(6)
    elif model == 'stereo_upconv_rnc':
        out_model = stereo_upconv_rnc(6)
    elif model == 'stereo_upconv_rnc_1':
        out_model = stereo_upconv_rnc_1(6)
    elif model == 'stereo_upconv_img_mask':
        out_model = stereo_upconv_img_mask(6)
    elif model == 'stereo_upconv_img_mask_f128':
        out_model = stereo_upconv_img_mask_f128(6)
    elif model == 'stereo_upconv_img_pt_mask':
        out_model = stereo_upconv_img_pt_mask(6)
    elif model == 'stereo_upconv_img_pt_mask_m128':
        out_model = stereo_upconv_img_pt_mask_m128(6)
    elif model == 'stereo_upconv_img_pt_mask_m128_v2':
        out_model = stereo_upconv_img_pt_mask_m128_v2(6)
    elif model == 'stereo_upconv_img_pt_mask_m256':
        out_model = stereo_upconv_img_pt_mask_m256(6)
    elif model == 'stereo_res_unet_upconv_crossclr2':
        out_model = stereo_res_unet_upconv_crossclr2(6)
    elif model == 'stereo_mask_res_mlp_crossclr2':
        out_model = stereo_mask_res_mlp_crossclr2(6)
    elif model == 'stereo_unet_crossclr2':
        out_model = stereo_unet_crossclr2(6)
    elif model == 'stereo_resnet18':
        out_model = stereo_resnet18(6)
    elif model == 'stereo_resnet50_mlp':
        out_model = stereo_resnet50_mlp(6)
    elif model == 'stereo_resnet18_mlp':
        out_model = stereo_resnet18_mlp(6)
    elif model == 'stereo_resnet50_mask_axis':
        out_model = stereo_resnet50_mask_axis(6)
    elif model == 'stereo_mask_disparity_res_mlp':
        out_model = stereo_mask_disparity_res_mlp(6)
    elif model == 'stereo_disparity_res_mlp':
        out_model = stereo_disparity_res_mlp(6)
    elif model == 'stereo_depth_axis_res_mlp':
        out_model = stereo_depth_axis_res_mlp(6)
    elif model == 'stereo_skresnet':
        out_model = stereo_skresnet(6)
    elif model == 'stereo_skresnet_depth':
        out_model = stereo_skresnet_depth(6)
    elif model == 'stereo_skresnet_axis':
        out_model = stereo_skresnet_axis(6)
    elif model == 'stereo_skresnet_axis_depth':
        out_model = stereo_skresnet_axis_depth(6)
    elif model == 'stereo_vit_axis':
        out_model = stereo_vit_axis(6)
    else:
        print('model selected does not exist')
    return out_model


def get_regressor(regressor_model, num_in_feature=1556, pretrained=False):
    out_regressor = regressor(num_in_feature)
    return out_regressor


def prepare_dataloader(data_directory, is_stereo, mode_flag, batch_size, num_workers, size, debug_mode=False, debug_size=100):
    datasets = [SENSEIDataset(data_directory, size, is_stereo, mode_flag, debug_mode, debug_size)]
    dataset = ConcatDataset(datasets)
    n_img = len(dataset)
    print('mode:', mode_flag, ': Use a dataset with ', n_img, 'images')
    if mode_flag == 'train':
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)
    return n_img, loader

def prepare_kfold_dataloader(data_directory, is_stereo, mode_flag, batch_size, num_workers, size, debug_mode=False, debug_size=100):
    datasets = [SENSEIDataset(data_directory, size, is_stereo, mode_flag, debug_mode, debug_size)]
    dataset = ConcatDataset(datasets)
    n_img = len(dataset)
    print('Use a dataset with ', n_img, 'images')
    return dataset, n_img


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def transform2Dto3D(Z, pt2d, alpha, beta, ox, oy):  # (u, v) is 2D laser point and changing for each image
    u, v = pt2d[0]
    X = (Z * (u - ox)) / alpha
    Y = (Z * (v - oy)) / beta

    return np.array([X, Y, Z])