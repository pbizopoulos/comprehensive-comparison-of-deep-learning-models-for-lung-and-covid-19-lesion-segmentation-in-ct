# Monkey patching start
def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script
# Monkey patching end

import numpy as np
import torch
from gooey import Gooey, GooeyParser
from matplotlib import pyplot as plt
from PIL import Image
from segmentation_models_pytorch import FPN as _FPN
from segmentation_models_pytorch import Linknet as _Linknet
from segmentation_models_pytorch import PSPNet as _PSPNet
from segmentation_models_pytorch import Unet as _Unet
from torchvision.transforms import functional as tf


def preprocess(image):
    if image.shape[0] in [1, 3]:
        image = image[0]
    elif image.shape[-1] in [1, 3]:
        image = image[..., -1]
    image = tf.to_pil_image(image)
    image = tf.resize(image, [512, 512])
    image = tf.to_tensor(image)
    image = tf.normalize(image, image.mean(), image.std())
    return image.unsqueeze(0)


def predict(image, experiment_name, architecture_name, encoder, encoder_weights):
    if architecture_name == 'Unet':
        architecture = _Unet
    if architecture_name == 'Linknet':
        architecture = _Linknet
    if architecture_name == 'FPN':
        architecture = _FPN
    if architecture_name == 'PSPNet':
        architecture = _PSPNet
    model = architecture(encoder, encoder_weights=encoder_weights, activation='sigmoid', in_channels=1).to('cpu')
    checkpoint = f'https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/releases/download/v1/{experiment_name}-{architecture_name}-{encoder}-{encoder_weights}.pt'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, map_location='cpu'))
    model.eval()
    preprocessed_image = preprocess(image)
    prediction = model(preprocessed_image)
    prediction = prediction[0, 0].detach().numpy()
    prediction = prediction > 0.5
    return prediction, preprocessed_image


@Gooey(program_name='CLI GUI Example App')
def main():
    experiment_name_list = ['lung-segmentation', 'lesion-segmentation-a']
    architecture_name_list = ['Unet', 'Linknet', 'FPN', 'PSPNet']
    encoder_list = ['vgg11', 'vgg13', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnext50_32x4d', 'dpn68', 'dpn98', 'mobilenet_v2', 'xception', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6']
    encoder_weights_list = [None, 'imagenet']

    parser = GooeyParser(description='v1')
    parser.add_argument('filename', widget="FileChooser", gooey_options={'default_dir': './', 'default_file': 'covid-19-pneumonia-4.jpg', 'wildcard': '*.jpg'})
    parser.add_argument('experiment_name', widget="Dropdown", choices=experiment_name_list, default='lung-segmentation')
    parser.add_argument('architecture_name', widget="Dropdown", choices=architecture_name_list, default='Unet')
    parser.add_argument('encoder', widget="Dropdown", choices=encoder_list, default='mobilenet_v2')
    parser.add_argument('encoder_weights', widget="Dropdown", choices=encoder_weights_list, default='imagenet')
    args = parser.parse_args()

    if args.filename:
        image = Image.open('./covid-19-pneumonia-4.jpg')
    else:
        image = Image.open('./covid-19-pneumonia-4.jpg')
    image = np.asarray(image)
    prediction, preprocessed_image = predict(image, args.experiment_name, args.architecture_name, args.encoder, args.encoder_weights)
    plt.subplot(121)
    plt.imshow(preprocessed_image[0, 0], cmap='gray')
    plt.subplot(122)
    plt.imshow(preprocessed_image[0, 0], cmap='gray')
    plt.imshow(prediction, cmap='Reds', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()
