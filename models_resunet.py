from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
# import tclib
from vit.vit_pytorch.vit import ViT


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                                   padding=int(np.floor((self.kernel_size - 1) / 2)))
        self.normalize = nn.BatchNorm2d(num_out_layers)
        self.elu = F.elu

    def forward(self, x):
        x = self.conv_base(x)
        x = self.normalize(x)
        x = self.elu(x)
        # print('x', x.shape)    # x torch.Size([5, 64, 115, 153])
        # return F.elu(x, inplace=True)
        return x

class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return F.max_pool2d((x), self.kernel_size, stride=2, padding=int(np.floor((self.kernel_size-1) / 2)))


class resconv_R(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_R, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*num_out_layers)
        self.elu = F.elu

    def forward(self, x):
        # do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        x = self.elu(self.normalize(x_out + shortcut))
        return x
        # return F.elu(self.normalize(x_out + shortcut), inplace=True)


class resconv_L(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_L, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*num_out_layers)
        self.elu = F.elu

    def forward(self, x):
        # do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        x = self.elu(self.normalize(x_out + shortcut))
        return x
        # return F.elu(self.normalize(x_out + shortcut), inplace=True)


class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)
        self.elu = F.elu

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        x = self.elu(self.normalize(x_out + shortcut))
        return x


def resblock_L(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_L(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv_L(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv_L(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_R(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_R(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv_R(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv_R(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        return x


class get_segmap(nn.Module):
    def __init__(self, num_in_layers):
        super(get_segmap, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 4, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.normalize = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.normalize(x)   # todo recover
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

class get_pt(nn.Module):
    def __init__(self, num_in_layers):
        super(get_pt, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 4, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.normalize = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

class stereo_res_unet(nn.Module):
    def __init__(self, num_in_layers):
        super(stereo_res_unet, self).__init__()
        # input image b x 6 x w x h (batch size, 6 channels, width, height)
        self.sigmoid = torch.nn.Sigmoid()
        # encoder
        # self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D
        self.conv1_L = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1_L = maxpool(3)  # H/4  -   64D
        self.conv2_L = resblock_L(64, 64, 3, 2)  # H/8  -  256D
        self.conv3_L = resblock_L(256, 128, 4, 2)  # H/16 -  512D
        self.conv4_L = resblock_L(512, 256, 6, 2)  # H/32 - 1024D
        # Bridge
        self.conv5_L = resblock_L(1024, 512, 3, 2)  # H/64 - 2048D

        # skip connection
        self.adaptive_pool = nn.AdaptiveMaxPool2d((15, 20))  # AdaptiveAvgPool2d
        self.adjust_conv5_L = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        self.adjust_conv4_L = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.adjust_conv3_L = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.adjust_conv2_L = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)

        # decoder
        self.iconv6_L = conv(2048, 512, 3, 1)
        self.iconv5_L = conv(512, 256, 3, 1)
        self.iconv4_L = conv(256, 128, 3, 1)
        self.iconv3_L = conv(128, 64, 3, 1)
        self.iconv2_L = conv(64, 32, 3, 1)
        # Full connection
        self.fc1 = nn.Linear(9600, 2128)  # 原来：(6272, 2128)
        self.fc2 = nn.Linear(2128, 532)

        # input points
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        # input the masks
        self.mask_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),  # H/2
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=2, padding=1),  # H/4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(140760, 512),
            nn.ReLU(),
        )

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.5)

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(532 + 512 + 512, 256),  # 现在总共有532（来自图像）+ 512（来自点特征）+ 512 （来自掩模）的特征
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points, masks):
        # input image b x 6 x w x h
        # Encoding
        x1_L = self.conv1_L(images) # encoder 1 H/2
        x_pool1_L = self.pool1_L(x1_L) # H/4
        x2_L = self.conv2_L(x_pool1_L) # encoder 2 H/8
        x3_L = self.conv3_L(x2_L)  # encoder 3 H/16
        x4_L = self.conv4_L(x3_L)  # encoder 4 H/32

        x5_L = self.conv5_L(x4_L) # Bridge

        # Decoding with skip connection
        iconv6_L = self.iconv6_L(x5_L)
        x5_L_adjusted = self.adjust_conv5_L(x5_L)
        iconv5_L = self.iconv5_L(iconv6_L + x5_L_adjusted)
        x4_L_adjusted = self.adjust_conv4_L(x4_L)
        x4_L_adjusted = self.adaptive_pool(x4_L_adjusted)
        iconv4_L = self.iconv4_L(iconv5_L + x4_L_adjusted)
        x3_L_adjusted = self.adjust_conv3_L(x3_L)
        x3_L_adjusted = self.adaptive_pool(x3_L_adjusted)
        iconv3_L = self.iconv3_L(iconv4_L + x3_L_adjusted)
        x2_L_adjusted = self.adjust_conv2_L(x2_L)
        x2_L_adjusted = self.adaptive_pool(x2_L_adjusted)
        iconv2_L = self.iconv2_L(iconv3_L + x2_L_adjusted)

        # Flatten and fully connected layers
        f_image = torch.flatten(iconv2_L, 1)
        f_image = self.fc1(f_image)
        f_image = self.fc2(f_image)
        f_image = self.dropout(f_image)

        # points
        pt_feature = self.layers(points.float())
        mask_feature = self.mask_layers(masks.float())

        fuse = torch.cat((f_image, pt_feature, mask_feature), 1)  # concatenate image, point and mask feature
        self.left_pred = self.decoder(fuse)  # use another MLP (decoder) to predict on the fused feature

        return self.left_pred

class stereo_res_unet_upconv(nn.Module):
    def __init__(self, num_in_layers):
        super(stereo_res_unet_upconv, self).__init__()
        # input image b x 6 x w x h (batch size, 6 channels, width, height)
        self.sigmoid = torch.nn.Sigmoid()
        # encoder
        # self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D
        self.conv1_L = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1_L = maxpool(3)  # H/4  -   64D
        self.conv2_L = resblock_L(64, 64, 3, 2)  # H/8  -  256D
        self.conv3_L = resblock_L(256, 128, 4, 2)  # H/16 -  512D
        self.conv4_L = resblock_L(512, 256, 6, 2)  # H/32 - 1024D
        # Bridge
        self.conv5_L = resblock_L(1024, 512, 3, 2)  # H/64 - 2048D

        # for skip connection
        self.adjust_conv5_L = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        self.adaptive_pool5 = nn.AdaptiveAvgPool2d((30, 40))
        self.adjust_conv4_L = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.adaptive_pool4 = nn.AdaptiveAvgPool2d((60, 80))
        self.adjust_conv3_L = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.adaptive_pool3 = nn.AdaptiveAvgPool2d((120, 160))
        self.adjust_conv2_L = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((240, 320))

        # decoder
        self.iconv6_L = conv(2048, 512, 3, 1)
        self.iconv5_L = conv(512, 256, 3, 1)
        self.iconv4_L = conv(256, 128, 3, 1)
        self.iconv3_L = conv(128, 64, 3, 1)
        self.iconv2_L = conv(64, 32, 3, 1)

        # upconv
        self.upconv6_L = upconv(512, 512, 3, scale=2)
        self.upconv5_L = upconv(256, 256, 3, scale=2)
        self.upconv4_L = upconv(128, 128, 3, scale=2)
        self.upconv3_L = upconv(64, 64, 3, scale=2)

        # Add a global average pooling layer
        self.global_avg_pool = nn.AdaptiveMaxPool2d((15, 20))  # AdaptiveAvgPool2d

        # Full connection
        self.fc1 = nn.Linear(9600, 2128)
        self.fc2 = nn.Linear(2128, 256)

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.5)

        # input points
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # input the masks
        self.mask_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),  # H/2
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=2, padding=1),  # H/4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(140760, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256+128+128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points, masks):
        # input image b x 6 x w x h
        # Encoding
        x1_L = self.conv1_L(images) # encoder 1 H/2
        x_pool1_L = self.pool1_L(x1_L) # H/4
        x2_L = self.conv2_L(x_pool1_L) # encoder 2 H/8
        x3_L = self.conv3_L(x2_L)  # encoder 3 H/16
        x4_L = self.conv4_L(x3_L)  # encoder 4 H/32

        x5_L = self.conv5_L(x4_L) # Bridge

        # Decoding with skip connection
        iconv6_L = self.iconv6_L(x5_L)
        up6_L = self.upconv6_L(iconv6_L)
        x5_L_adjusted = self.adjust_conv5_L(x5_L)
        x5_L_adjusted = self.adaptive_pool5(x5_L_adjusted)
        iconv5_L = self.iconv5_L(up6_L + x5_L_adjusted)
        up5_L = self.upconv5_L(iconv5_L)
        x4_L_adjusted = self.adjust_conv4_L(x4_L)
        x4_L_adjusted = self.adaptive_pool4(x4_L_adjusted)
        iconv4_L = self.iconv4_L(up5_L + x4_L_adjusted)
        up4_L = self.upconv4_L(iconv4_L)
        x3_L_adjusted = self.adjust_conv3_L(x3_L)
        x3_L_adjusted = self.adaptive_pool3(x3_L_adjusted)
        iconv3_L = self.iconv3_L(up4_L + x3_L_adjusted)
        up3_L = self.upconv3_L(iconv3_L)
        x2_L_adjusted = self.adjust_conv2_L(x2_L)
        x2_L_adjusted = self.adaptive_pool2(x2_L_adjusted)
        iconv2_L = self.iconv2_L(up3_L + x2_L_adjusted)

        # Flatten and fully connected layers
        pooled_output = self.global_avg_pool(iconv2_L)
        f_image = torch.flatten(pooled_output, 1)
        f_image = self.fc1(f_image)
        f_image = self.fc2(f_image)
        f_image = self.dropout(f_image)

        # points
        pt_feature = self.layers(points.float())
        mask_feature = self.mask_layers(masks.float())

        fuse = torch.cat((f_image, pt_feature, mask_feature), 1)  # concatenate image, point and mask feature
        self.left_pred = self.decoder(fuse)  # use another MLP (decoder) to predict on the fused feature

        return self.left_pred


class stereo_res_unet_upconv_img(nn.Module):
    def __init__(self, num_in_layers):
        super(stereo_res_unet_upconv_img, self).__init__()
        # input image b x 6 x w x h (batch size, 6 channels, width, height)
        self.sigmoid = torch.nn.Sigmoid()
        # encoder
        # self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D
        self.conv1_L = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1_L = maxpool(3)  # H/4  -   64D
        self.conv2_L = resblock_L(64, 64, 3, 2)  # H/8  -  256D
        self.conv3_L = resblock_L(256, 128, 4, 2)  # H/16 -  512D
        self.conv4_L = resblock_L(512, 256, 6, 2)  # H/32 - 1024D
        # Bridge
        self.conv5_L = resblock_L(1024, 512, 3, 2)  # H/64 - 2048D

        # for skip connection
        self.adjust_conv5_L = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        self.adaptive_pool5 = nn.AdaptiveAvgPool2d((30, 40))
        self.adjust_conv4_L = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.adaptive_pool4 = nn.AdaptiveAvgPool2d((60, 80))
        self.adjust_conv3_L = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.adaptive_pool3 = nn.AdaptiveAvgPool2d((120, 160))
        self.adjust_conv2_L = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((240, 320))

        # decoder
        self.iconv6_L = conv(2048, 512, 3, 1)
        self.iconv5_L = conv(512, 256, 3, 1)
        self.iconv4_L = conv(256, 128, 3, 1)
        self.iconv3_L = conv(128, 64, 3, 1)
        self.iconv2_L = conv(64, 32, 3, 1)

        # upconv
        self.upconv6_L = upconv(512, 512, 3, scale=2)
        self.upconv5_L = upconv(256, 256, 3, scale=2)
        self.upconv4_L = upconv(128, 128, 3, scale=2)
        self.upconv3_L = upconv(64, 64, 3, scale=2)

        # Add a global average pooling layer
        self.global_avg_pool = nn.AdaptiveMaxPool2d((15, 20))  # AdaptiveAvgPool2d

        # Full connection
        self.fc1 = nn.Linear(9600, 2128)
        self.fc2 = nn.Linear(2128, 512)

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.5)

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images):
        # input image b x 6 x w x h
        # Encoding
        x1_L = self.conv1_L(images) # encoder 1 H/2
        x_pool1_L = self.pool1_L(x1_L) # H/4
        x2_L = self.conv2_L(x_pool1_L) # encoder 2 H/8
        x3_L = self.conv3_L(x2_L)  # encoder 3 H/16
        x4_L = self.conv4_L(x3_L)  # encoder 4 H/32

        x5_L = self.conv5_L(x4_L) # Bridge

        # Decoding with skip connection
        iconv6_L = self.iconv6_L(x5_L)
        up6_L = self.upconv6_L(iconv6_L)
        x5_L_adjusted = self.adjust_conv5_L(x5_L)
        x5_L_adjusted = self.adaptive_pool5(x5_L_adjusted)

        print("up6_L size:", up6_L.size())
        print("x5_L_adjusted size:", x5_L_adjusted.size())

        iconv5_L = self.iconv5_L(up6_L + x5_L_adjusted)
        up5_L = self.upconv5_L(iconv5_L)
        x4_L_adjusted = self.adjust_conv4_L(x4_L)
        x4_L_adjusted = self.adaptive_pool4(x4_L_adjusted)
        iconv4_L = self.iconv4_L(up5_L + x4_L_adjusted)
        up4_L = self.upconv4_L(iconv4_L)
        x3_L_adjusted = self.adjust_conv3_L(x3_L)
        x3_L_adjusted = self.adaptive_pool3(x3_L_adjusted)
        iconv3_L = self.iconv3_L(up4_L + x3_L_adjusted)
        up3_L = self.upconv3_L(iconv3_L)
        x2_L_adjusted = self.adjust_conv2_L(x2_L)
        x2_L_adjusted = self.adaptive_pool2(x2_L_adjusted)
        iconv2_L = self.iconv2_L(up3_L + x2_L_adjusted)

        # Flatten and fully connected layers
        pooled_output = self.global_avg_pool(iconv2_L)
        f_image = torch.flatten(pooled_output, 1)  # fully connected layers need 1D input
        f_image = self.fc1(f_image)
        f_image = self.fc2(f_image)
        f_image = self.dropout(f_image)

        # 合并图像特征、点特征和掩模特征
        self.left_pred = self.decoder(f_image)  # use another MLP (decoder) to predict on the fused feature

        return self.left_pred


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


