# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import time
import logging

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.logger = logging.getLogger("maskrcnn_benchmark.generalized_rcnn")
        self.meters = MetricLogger(delimiter="  ")
        self._iter = 0

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        backbone_time = time.time()
        features = self.backbone(images.tensors)
        backbone_time = time.time() - backbone_time

        rpn_time = time.time()
        proposals, proposal_losses = self.rpn(images, features, targets)
        rpn_time = time.time() - rpn_time
        if self.roi_heads:
            roi_time = time.time()
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
            roi_time = time.time() - roi_time
            self.meters.update(roi=roi_time)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        # TODO: logging time
        self._iter += 1
        self.meters.update(backbone=backbone_time, rpn=rpn_time)
        if self._iter % 20 == 0:
            self.logger.info("TIME: {}".format(str(self.meters)))

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
