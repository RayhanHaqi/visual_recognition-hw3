import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


PARAM_BUDGET = 200_000_000


def build_maskrcnn(num_classes=5, pretrained=True, min_size=512, max_size=1024,
                   anchor_sizes=None, box_detections_per_img=100):
    weights = "DEFAULT" if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        box_detections_per_img=box_detections_per_img,
    )
    if anchor_sizes is not None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable >= PARAM_BUDGET:
        raise ValueError(f"Trainable params {n_trainable/1e6:.1f}M exceed 200M course budget.")
    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
