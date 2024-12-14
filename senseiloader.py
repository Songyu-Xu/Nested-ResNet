import torch
import torch.nn
import numpy as np
import os
import os.path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import imageio
import torch.nn.functional as F_torch


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class SENSEIDataset(torch.utils.data.Dataset):
    def __init__(self, directory, size, is_stereo=True, mode_flag='train', debug_mode=False, debug_size=100):
        '''
        directory is expected to contain some folder structure:
        '''
        super().__init__()
        self.img_ext = '.jpg'
        self.gt_ext = '.npy'
        self.point_ext = '.txt'
        self.loader = pil_loader
        self.mode_flag = mode_flag
        self.is_stereo = is_stereo
        self.directory = directory
        self.size = size
        self.gt_split_line = self.readlines('./coffbea-2023/data/split/' + mode_flag + '.txt')
        # self.gt_split_line = self.readlines('./coffbea-2023/data/3d_ground_truth/' + mode_flag + '.txt')

        # k-fold
        # if self.mode_flag == 'test':
        #     self.gt_split_line = self.readlines('./coffbea-2023/data/3d_ground_truth/test.txt')
        # else:
        #     self.gt_split_line = self.readlines('./coffbea-2023/data/3d_ground_truth/train_and_val.txt')

        # Debug mode adjustments
        if debug_mode:
            self.gt_split_line = self.gt_split_line[:debug_size]  # Use only a subset for debugging

    def __len__(self):
        return len(self.gt_split_line)

    def __getitem__(self, idx):
        '''
        points: points on the arrow
        '''
        # load images
        fileidx = self.gt_split_line[idx].split(',')[0]
        left_rgb, padding_left_left, padding_top_left, padding_right_left, padding_bottom_left = (
            self.data_transform(self.get_color(self.directory + '/camera1/rgb/' + fileidx), self.mode_flag, self.size))
        images = left_rgb
        if self.is_stereo:
            right_rgb, padding_left_right, padding_top_right, padding_right_right, padding_bottom_right = (
                self.data_transform(self.get_color(self.directory + '/camera0/rgb/' + fileidx), self.mode_flag, self.size))
            images = torch.cat((left_rgb, right_rgb), 0)

        points = self.readlines(self.directory + '/camera0/line_annotation_sample/' + fileidx.split('.')[0] + self.point_ext)  # 0 left, gt
        points_line = []
        for line in points:
            points_line.append([float(line.split(' ')[0]), float(line.split(' ')[1])])

        points = torch.tensor(points_line)

        # load center laser location
        line = self.gt_split_line[idx]
        line = line.strip().split(',')
        # Read x, y from the line
        label = torch.tensor([float(line[1]), float(line[2])]).float()
        # adjust the gt due to padding
        label = label + torch.tensor([padding_left_left, padding_top_left]).float()

        # Read x, y, and depth (z) from the line
        # label = torch.tensor([float(line[1]), float(line[2]), float(line[3])]).float()
        # # Adjust the x and y coordinates for padding, but depth (z) doesn't require adjustment
        # label[:2] = label[:2] + torch.tensor([padding_left_left, padding_top_left]).float()

        # load the depth GT for eval on coffbea for flr
        if self.mode_flag == 'test':
            depthgtroot = self.directory + '/camera0/depthGT/' + fileidx.split('.')[0] + '.npy'
            depthgt = np.load(depthgtroot)

        # mask
        left_probe_mask_root = self.directory + '/camera1/sensei_mask/' + fileidx.split('.')[0] + '.npy'
        left_probe_mask = torch.from_numpy(np.load(left_probe_mask_root)).unsqueeze(0)  # [1, H, W]
        pad = (padding_left_left, padding_right_left, padding_top_left, padding_bottom_left)
        left_probe_mask = F.pad(left_probe_mask, pad, padding_mode='constant', fill=0)
        probe_masks = left_probe_mask
        if self.is_stereo:
            right_probe_mask_root = self.directory + '/camera0/sensei_mask/' + fileidx.split('.')[0] + '.npy'
            right_probe_mask = torch.from_numpy(np.load(right_probe_mask_root)).unsqueeze(0)
            pad = (padding_left_right, padding_right_right, padding_top_right, padding_bottom_right)
            right_probe_mask = F.pad(right_probe_mask, pad, padding_mode='constant', fill=0)
            probe_masks = torch.cat((left_probe_mask, right_probe_mask), 0)

        # load depth map from pfm
        depth_root = self.directory + '/depth_estimate/' + fileidx.split('.')[0] + '_depth.pfm'
        # depth_map = np.load(depth_root).astype(np.float32)
        # depth_map = torch.from_numpy(depth_map)
        # depth_map = F_torch.pad(depth_map, pad, value=0, mode='constant')
        depth_map = imageio.imread(depth_root, format='PFM')
        depth_map = torch.from_numpy(depth_map).unsqueeze(0)
        depth_map = F.pad(depth_map, pad, fill=0, padding_mode='constant')

        if self.mode_flag == 'test':
            sample = {'images': images, 'points': points, 'labels': label, 'depthgt': depthgt,
                      'probe_masks': probe_masks, 'padding_left': padding_left_left, 'padding_top': padding_top_left,
                      'depth_map': depth_map}
        else:
            sample = {'images': images, 'points': points, 'labels': label,
                      'probe_masks': probe_masks, 'padding_left': padding_left_left, 'padding_top': padding_top_left,
                      'depth_map': depth_map}

        return sample

    def get_color(self, img_path):
        color = self.loader(img_path)
        return color

    def data_transform(self, input, mode_flag, size):
        width, height = input.size
        desired_size = size[0]
        # padded pixels
        delta_w = desired_size - width
        delta_h = desired_size - height
        padding_left = delta_w // 2
        padding_right = delta_w - padding_left
        padding_top = delta_h // 2
        padding_bottom = delta_h - padding_top

        # padding
        input = F.pad(input, [padding_left, padding_top, padding_right, padding_bottom], fill=0,
                      padding_mode='constant')

        if mode_flag == 'train':
            trans = transforms.Compose([
                # ResizeImage(train=True, size=size),
                # RandomCrop(train=True, size=size),
                # Crop(size),
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.Compose([
                # ResizeImage(size=size, train=False),
                # RandomCrop(size=size, train=False),
                # CenterCrop(size),
                transforms.ToTensor(),
            ])

        return trans(input), padding_left, padding_top, padding_right, padding_bottom

    def readlines(self, filename):
        """Read all the lines in a text file and return as a list
        """
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return lines


class ResizeImage(object):
    def __init__(self, size, train=True):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        sample = self.transform(sample)
        return sample


class CenterCrop(object):
    def __init__(self, size, train=True):
        self.train = train
        self.transform = transforms.CenterCrop(size)

    def __call__(self, sample):
        sample = self.transform(sample)
        return sample



