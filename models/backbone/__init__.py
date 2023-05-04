from models.backbone import resnet


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm)
    else:
        raise NotImplementedError
