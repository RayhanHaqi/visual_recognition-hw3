import timm
import torch
import torchvision
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


PARAM_BUDGET = 200_000_000
CONVNEXT_MODEL_NAME = "convnext_base.fb_in22k_ft_in1k_384"


class ConvNeXtBackbone(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = timm.create_model(CONVNEXT_MODEL_NAME, pretrained=pretrained, features_only=False)
        self.stem = m.stem
        self.s0 = m.stages[0]
        self.s1 = m.stages[1]
        self.s2 = m.stages[2]
        self.s3 = m.stages[3]
        self.out_channels = [128, 256, 512, 1024]


def _build_convnext_fpn(pretrained=True):
    body = ConvNeXtBackbone(pretrained=pretrained)
    return BackboneWithFPN(
        body,
        return_layers={"s0": "0", "s1": "1", "s2": "2", "s3": "3"},
        in_channels_list=body.out_channels,
        out_channels=256,
        extra_blocks=None,
    )


def _transfer_coco_heads(target_model):
    src = maskrcnn_resnet50_fpn_v2(weights="DEFAULT", num_classes=91)
    src_state = src.state_dict()
    target_state = target_model.state_dict()

    transferred = 0
    for key in target_state:
        if not key.startswith("backbone") and key in src_state:
            src_shape = src_state[key].shape
            tgt_shape = target_state[key].shape
            if src_shape == tgt_shape:
                target_state[key] = src_state[key]
                transferred += 1

    target_model.load_state_dict(target_state)
    del src
    print(f"  Transferred {transferred} head params from COCO-pretrained ResNet-50.")


def build_maskrcnn(num_classes=5, pretrained=True, min_size=512, max_size=1024,
                   anchor_sizes=None, box_detections_per_img=100, backbone="resnet50"):
    if backbone == "resnet50":
        weights = "DEFAULT" if pretrained else None
        model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            min_size=min_size,
            max_size=max_size,
            box_detections_per_img=box_detections_per_img,
        )
    elif backbone == "convnext_base":
        bb = _build_convnext_fpn(pretrained=pretrained)
        model = MaskRCNN(
            bb,
            num_classes=91 if pretrained else num_classes,
            min_size=min_size,
            max_size=max_size,
            box_detections_per_img=box_detections_per_img,
        )
        if pretrained:
            _transfer_coco_heads(model)
        if not pretrained:
            model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2,
            )
            model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2,
            )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

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
