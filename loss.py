import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class TwinLoss(nn.modules.Module):
    def __init__(self, L1_w=0.5, L2_w=0.5, L_left_w=0.5, L_right_w=0.5, att_w=0.5):
        super(TwinLoss, self).__init__()
        self.L1_w = L1_w
        self.L2_w = L2_w
        self.L_left_w = L_left_w
        self.L_right_w = L_right_w
        self.att_w = att_w

        self.loss = nn.MSELoss()
        self.att_loss_func = nn.MSELoss()

    def BCE_Loss(self, pred, gt):
        BCE_loss = nn.BCELoss()
        loss = BCE_loss(pred, gt)
        return loss

    def Attention_Loss(self, positive_att_map, body_mask):
        """
        Args:
        positive_att_map - positive attention map which directs focus on the probe
        body_mask - probe mask
        """
        return self.att_loss_func(positive_att_map, body_mask)

    def forward(self, left_pred, left_gt, positive_att_map=None, att_mask=None):
        """
        Args:xx
        Return:
            (float): The loss
        """

        # left_loss = self.L1_w * self.L1_loss(left_pred, left_gt) + self.L2_w * self.L2_loss(right_pred, right_gt)
        # right_loss = self.L1_w * self.L1_loss(right_pred, right_gt) + self.L2_w * self.L2_loss(right_pred, right_gt)
        # loss = self.L_right_w * left_loss + self.L_right_w * right_loss

        # loss = self.L1_w * self.BCE_Loss(left_pred, left_gt) + self.L2_w * self.BCE_Loss(right_pred, right_gt)

        # x_pred, y_pred, z_pred = left_pred[:, 0], left_pred[:, 1], left_pred[:, 2]
        # x_gt, y_gt, z_gt = left_gt[:, 0], left_gt[:, 1], left_gt[:, 2]
        # scale_factor = 0.35
        # z_pred_scaled = z_pred * scale_factor
        # z_gt_scaled = z_gt * scale_factor
        # left_pred_scaled = torch.stack([x_pred, y_pred, z_pred_scaled], dim=1)
        # left_gt_scaled = torch.stack([x_gt, y_gt, z_gt_scaled], dim=1)
        # loss = self.loss(left_pred_scaled, left_gt_scaled)

        loss = self.loss(left_pred, left_gt)

        return loss

