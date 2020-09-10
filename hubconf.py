dependencies = ['segmentation_models_pytorch']

import torch
from segmentation_models_pytorch import Unet, Linknet, FPN, PSPNet


def segmentation_model(pretrained=True, experiment_name='lung-segmentation', architecture_name='Unet', encoder='resnet18', encoder_weights='imagenet'):
    """
    segmentation model
    pretrained (bool): load pretrained weights into the model
    experiment_name ('lung-segmentation', 'lesion-segmentation-a'): Experiment name
    architecture_name ('Unet', 'Linknet', 'FPN', 'PSPNet'): Architecture name
    encoder ('vgg11', 'vgg13', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnext50_32x4d', 'dpn68', 'dpn98', 'mobilenet_v2', 'xception', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6'): Encoder
    encoder_weights (None, 'imagenet'): Encoder weights
    """
    if architecture_name == 'Unet':
        architecture = Unet
    if architecture_name == 'Linknet':
        architecture = Linknet
    if architecture_name == 'FPN':
        architecture = FPN
    if architecture_name == 'PSPNet':
        architecture = PSPNet
    model = architecture(encoder, encoder_weights=encoder_weights, activation='sigmoid', in_channels=1).to('cpu')
    if pretrained:
        checkpoint = f'https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/releases/download/v1/{experiment_name}-{architecture_name}-{encoder}-{encoder_weights}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location='cpu'))
    return model
