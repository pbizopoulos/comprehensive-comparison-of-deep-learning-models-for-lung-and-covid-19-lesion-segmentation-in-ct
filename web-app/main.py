import gradio
import torch

from matplotlib import pyplot as plt
from torchvision.transforms import functional as tf

from segmentation_models_pytorch import Unet, Linknet, FPN, PSPNet


experiment_list = ['Lung segmentation', 'Lesion segmentation A']
experiment_name_list = [experiment.lower().replace(' ', '-') for experiment in experiment_list]
architecture_list = [Unet, Linknet, FPN, PSPNet]
architecture_name_list = [architecture.__name__ for architecture in architecture_list]
encoder_list = ['vgg11', 'vgg13', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnext50_32x4d', 'dpn68', 'dpn98', 'mobilenet_v2', 'xception', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6']
encoder_weights_list = ['None', 'imagenet']

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
        architecture = Unet
    if architecture_name == 'Linknet':
        architecture = Linknet
    if architecture_name == 'FPN':
        architecture = FPN
    if architecture_name == 'PSPNet':
        architecture = PSPNet
    model = architecture(encoder, encoder_weights=encoder_weights, activation='sigmoid', in_channels=1).to('cpu')
    checkpoint = f'https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/releases/latest/download/{experiment_name}-{architecture_name}-{encoder}-{encoder_weights}.pt'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location='cpu'))
    model.eval()
    image = preprocess(image)
    prediction = model(image)
    prediction = prediction[0, 0].detach().numpy()
    prediction = prediction > 0.5
    fig, ax = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(image[0, 0], cmap='gray', vmin=-1.5, vmax=1.5)
    plt.imshow(prediction, cmap='Reds', alpha=0.5)
    return plt


if __name__ == '__main__':
    gradio.reset_all()
    gradio.Interface(predict, 
            [
                gradio.inputs.Image(shape=(512, 512, 1)),
                gradio.inputs.Dropdown(experiment_name_list),
                gradio.inputs.Dropdown(architecture_name_list),
                gradio.inputs.Dropdown(encoder_list),
                gradio.inputs.Dropdown(encoder_weights_list),
                ], 
            [
                gradio.outputs.Image(plot=True),
                ],
            title='COVID-19 lung/lesion segmentation',
            description='DISCLAIMER: THIS TOOL IS FOR RESEARCH USE ONLY AND IT IS NOT INTENDED TO BE USED FOR MEDICAL PURPOSES. Supplementary User Interface for the paper "Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT scans"',
            examples=[
                ['covid-19-pneumonia-4.jpg', 'lung-segmentation', 'PSPNet', 'mobilenet_v2', 'imagenet'],
                ['covid-19-pneumonia-45.jpg', 'lesion-segmentation-a', 'PSPNet', 'vgg11', 'imagenet'],
                ],
            server_name='0.0.0.0'
            ).launch()
