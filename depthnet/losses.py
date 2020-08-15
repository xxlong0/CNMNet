import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IdepthLoss_234(nn.Module):
    def forward(self, idepth_preds, idepth_ground_truth):
        """
        Calculate the inverse depth loss of disp2, disp3, and disp4
        :param idepth_preds: a list of [disp1, disp2, disp3, disp4] idepth_pred in different scales
        :param idepth_ground_truth: The ground truth of idepth, which should have same size with disp1
        :return: the loss of idepth
        """
        batch_size = idepth_ground_truth.shape[0]
        [disp1, disp2, disp3, disp4] = idepth_preds
        disp1_max = disp1.max()
        idepth_ground_truth_2 = F.interpolate(idepth_ground_truth, size=disp2.shape[2:4])
        idepth_ground_truth_3 = F.interpolate(idepth_ground_truth, size=disp3.shape[2:4])
        idepth_ground_truth_4 = F.interpolate(idepth_ground_truth, size=disp4.shape[2:4])

        # loss1 = (disp1 - idepth_ground_truth).abs().mean()
        loss2 = (disp2 - idepth_ground_truth_2).abs().mean()
        loss3 = (disp3 - idepth_ground_truth_3).abs().mean()
        loss4 = (disp4 - idepth_ground_truth_4).abs().mean()

        return 0.1 * (loss2 + loss3 + loss4) / 3.0


class IdepthLoss(nn.Module):
    def forward(self, idepth_pred, idepth_groud_truth, log=False):
        """
        Calculate the inverse depth loss of disp1 (same size with ground truth)
        :param idepth_pred: [b, 1, h, w]
        :param idepth_groud_truth:
        :return:
        """
        b, c, h, w = idepth_pred.shape
        mask = (idepth_groud_truth > 0.0) & (torch.isfinite(idepth_groud_truth)) & (torch.isfinite(idepth_pred)) & (
                    idepth_pred > 0.0)

        idepth_pred_mask = idepth_pred[mask]
        idepth_groud_truth_mask = idepth_groud_truth[mask]

        if log:
            return F.l1_loss(torch.log10(idepth_pred_mask), torch.log10(idepth_groud_truth_mask))
        else:
            return F.l1_loss(idepth_pred_mask, idepth_groud_truth_mask)


class IdepthwithProbLoss(nn.Module):
    def forward(self, idepth_pred, idepth_gt, prob_map, log=False):
        """
        Calculate the idepth loss with prob-map
        :param idepth_pred: [b, 1, h, w]
        :param idepth_gt:
        :param prob_map: The probability map which indicates the reliability of depth predicted on one pixel
        :return:
        """
        b, _, h, w = idepth_gt.shape
        mask = (idepth_gt > 0.0) & (torch.isfinite(idepth_gt)) & (torch.isfinite(idepth_pred)) & (idepth_pred > 0.0)
        idepth_pred_mask = idepth_pred[mask]
        idepth_groud_truth_mask = idepth_gt[mask]
        prob_map_mask = prob_map[mask]

        if log:
            diff = 10 * torch.abs(torch.log10(idepth_pred_mask) - torch.log10(idepth_groud_truth_mask))
        else:
            diff = torch.abs(idepth_pred_mask - idepth_groud_truth_mask)
        loss = prob_map_mask * diff
        loss = loss.mean()

        return loss


def surface_normal_loss(prediction, surface_normal, valid_region, probability_map=None):
    """
    Calculate the loss of gt_normal and pred_normal
    :param prediction: [1, 3, h, w]
    :param surface_normal: [1, 3, h, w]
    :param valid_region: [1, 1, h, w]
    :return:
    """
    b, c, h, w = prediction.size()

    # surface normal may have NaN values
    map = (torch.isfinite(torch.sum(surface_normal, dim=1, keepdim=True))) & \
          (torch.isfinite(torch.sum(prediction, dim=1, keepdim=True)))

    valid_map = map.repeat((1, c, 1, 1))
    valid_region = valid_map & valid_region

    ## change to [c,b,h,w]
    prediction = prediction.permute(1, 0, 2, 3)
    surface_normal = surface_normal.permute(1, 0, 2, 3)

    if valid_region is None:
        valid_predition = torch.transpose(prediction.view(c, -1), 0, 1)
        valid_surface_normal = torch.transpose(surface_normal.view(c, -1), 0, 1)
        if probability_map is not None:
            probability_map = probability_map.permute(1, 0, 2, 3)
            valid_prob_map = torch.transpose(probability_map.view(1, -1), 0, 1)
    else:
        valid_region = valid_region.permute(1, 0, 2, 3)
        valid_predition = torch.transpose(torch.masked_select(prediction, valid_region)
                                          .view(c, -1), 0, 1)
        valid_surface_normal = torch.transpose(
            torch.masked_select(surface_normal, valid_region).view(c, -1), 0, 1)
        if probability_map is not None:
            probability_map = probability_map.permute(1, 0, 2, 3)
            valid_prob_map = torch.transpose(
                torch.masked_select(probability_map, valid_region[0:1, :, :, :]).view(1, -1), 0, 1).squeeze(-1)

    similarity = torch.nn.functional.cosine_similarity(valid_predition, valid_surface_normal, dim=1)

    if probability_map is None:
        loss = torch.mean(1 - similarity)
    else:
        loss = torch.sum((1 - similarity) * valid_prob_map) / (torch.sum(valid_prob_map))

    mean_angle = torch.mean(torch.acos(torch.clamp(similarity, -1, 1)))
    return loss, mean_angle / np.pi * 180
