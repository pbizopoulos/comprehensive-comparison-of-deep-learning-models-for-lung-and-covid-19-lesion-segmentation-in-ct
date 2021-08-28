import argparse
import glob
import itertools
import os
from zipfile import ZipFile

import gdown
import nibabel as nib
import numpy as np
import pandas as pd
import requests
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from segmentation_models_pytorch import FPN, Linknet, PSPNet, Unet
from segmentation_models_pytorch.utils.losses import DiceLoss
from skimage.measure import marching_cubes
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tf

plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'

eps = 1e-6


def metrics(prediction, mask):
    prediction = prediction > 0.5
    true_positive = (mask * prediction).sum().float()
    true_negative = (~mask * ~prediction).sum().float()
    false_positive = (mask * ~prediction).sum().float()
    false_negative = (~mask * prediction).sum().float()
    is_mask_and_prediction_empty = (true_positive + false_positive) == 0
    if is_mask_and_prediction_empty:
        specificity = 1
        sensitivity = 1
        dice = 1
    else:
        specificity = (true_negative / (true_negative + false_positive + eps)).item()
        sensitivity = (true_positive / (true_positive + false_negative + eps)).item()
        dice = (2 * true_positive / (2 * true_positive + false_positive + false_negative + eps)).item()
    return sensitivity, specificity, dice


def save_weights(model, experiment_name, architecture_name, encoder_weights):
    rows = 8
    columns = 8
    plt.figure(figsize=(4, 4.6))
    gs = gridspec.GridSpec(rows, columns, wspace=0.0, hspace=0.0, left=0)
    for index_rows in range(rows):
        for index_columns in range(columns):
            weight = list(model.children())[0].conv1.weight[index_rows * rows + index_columns]
            ax = plt.subplot(gs[index_rows, index_columns])
            plt.imshow(weight[0].detach().cpu().numpy(), cmap='gray', vmin=-0.4, vmax=0.4)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.savefig(f'tmp/{experiment_name}-{architecture_name}-{encoder_weights}-weights')
    plt.close()


def save_hist(hist_images, hist_masks, hist_range, experiment_name):
    t = np.linspace(hist_range[0], hist_range[1], hist_masks.shape[-1])
    hist_images = hist_images.reshape(-1, hist_images.shape[-1]).sum(0)
    hist_masks = hist_masks.reshape(-1, hist_masks.shape[-1]).sum(0)
    hist_maxes = max([hist_images.max(), hist_masks.max()])
    hist_images /= hist_maxes
    hist_masks /= hist_maxes
    _, ax = plt.subplots()
    plt.bar(t, hist_images, width=t[1] - t[0], align='center', label='Images')
    plt.bar(t, hist_masks, width=t[1] - t[0], align='center', label='Masks')
    plt.xlabel('Normalized values', fontsize=15)
    plt.xlim(hist_range)
    plt.ylim([10 ** (-7), 1])
    plt.grid(True, which='both')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(f'tmp/{experiment_name}-hist')
    plt.close()


def save_loss(loss, train_or_validation, experiment_name, architecture_name_list, ylim):
    loss = np.nan_to_num(loss)
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    p1 = [None] * len(architecture_name_list)
    p2 = [None] * len(architecture_name_list)
    t = np.arange(1, loss.shape[-1] + 1)
    loss = loss.reshape(loss.shape[0], -1, loss.shape[-1])
    mus = loss.mean(1)
    sigmas = loss.std(1)
    _, ax = plt.subplots()
    for index, (mu, sigma, color) in enumerate(zip(mus, sigmas, color_list)):
        ax.fill_between(t, mu + sigma, mu - sigma, facecolor=color, alpha=0.3)
        p1[index] = ax.plot(t, mu, color=color)
        p2[index] = ax.fill(np.nan, np.nan, color, alpha=0.3)
    ax.legend([(p2[0][0], p1[0][0]), (p2[1][0], p1[1][0]), (p2[2][0], p1[2][0]), (p2[3][0], p1[3][0])], architecture_name_list, loc='upper right')
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim(ylim)
    if train_or_validation not in ['Train', 'Validation']:
        plt.xlabel('Epochs', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize='large')
    ax.tick_params(axis='both', which='minor', labelsize='large')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f'tmp/{experiment_name}-{train_or_validation.lower().replace(" ", "-")}-loss')
    plt.close()


def save_architecture_box(dice, architecture_name_list, experiment_name):
    dice_ = dice.reshape(dice.shape[0], -1).T
    plt.subplots()
    plt.boxplot(dice_)
    plt.grid(True)
    plt.xticks(ticks=np.arange(len(architecture_name_list)) + 1, labels=architecture_name_list, fontsize=15)
    plt.ylim([70, 100])
    plt.savefig(f'tmp/{experiment_name}-boxplot-dice')
    plt.close()


def save_initialization_box(dice, encoder_weights_list):
    dice_ = dice.reshape(-1, dice.shape[-1])
    plt.subplots()
    plt.boxplot(dice_)
    plt.grid(True)
    plt.xticks(ticks=np.arange(len(encoder_weights_list)) + 1, labels=[str(e) for e in encoder_weights_list], fontsize=15)
    plt.ylim([70, 100])
    plt.savefig('tmp/initialization-boxplot-dice')
    plt.close()


def save_scatter(num_parameters_array, dice, architecture_name_list, experiment_name, ylim):
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    dice_ = dice.mean(-1)
    _, ax = plt.subplots()
    for num_parameters, d, architecture_name, color in zip(num_parameters_array, dice_, architecture_name_list, color_list):
        plt.scatter(num_parameters, d, c=color, label=architecture_name, s=3)
    x = num_parameters_array.flatten()
    y = dice_.flatten()
    xmin = 0
    xmax = 80
    ymin = ylim[0]
    ymax = ylim[1]
    nbins = 100
    X, Y = np.mgrid[xmin: xmax: nbins * 1j, ymin: ymax: nbins * 1j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.imshow(np.rot90(Z), cmap='Greens', extent=[xmin, xmax, ymin, ymax], alpha=0.5)
    plt.grid(True)
    plt.xlabel(r'Number of parameters ($10^6$)', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize='large')
    ax.tick_params(axis='both', which='minor', labelsize='large')
    plt.xlim([xmin, xmax])
    plt.ylim(ylim)
    ax.legend(loc='lower right')
    ax.set_aspect(aspect='auto')
    plt.savefig(f'tmp/{experiment_name}-scatter-dice-vs-num-parameters')
    plt.close()


def save_3d(volume, is_full, experiment_name, architecture, encoder_weights):
    if is_full:
        step_size = 1
    else:
        step_size = 10
    volume = volume > 0.5
    volume[0, 0, 0:10] = 0
    volume[0, 0, 10:20] = 1
    verts, faces, _, _ = marching_cubes(volume, 0.5, step_size=step_size)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.5, rasterized=True)
    ax.set_xlim([1, volume.shape[0]])
    ax.set_ylim([1, volume.shape[1]])
    ax.set_zlim([1, volume.shape[2]])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    plt.savefig(f'tmp/{experiment_name}-{architecture}-{encoder_weights}-volume')
    plt.close()


def save_image(image, experiment_name):
    image = image.cpu().numpy()
    _, ax = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(image, cmap='gray', vmin=-1.5, vmax=1.5)
    plt.savefig(f'tmp/{experiment_name}-image')
    plt.close()


def save_masked_image(image, mask, prediction, experiment_name, architecture, encoder):
    image = image.cpu().numpy()
    mask = mask.cpu().numpy()
    prediction = prediction.cpu().detach().numpy()
    prediction = prediction > 0.5
    correct = mask * prediction
    false = np.logical_xor(mask, prediction) > 0.5
    _, ax = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(image, cmap='gray', vmin=-1.5, vmax=1.5)
    plt.imshow(correct, cmap='Greens', alpha=0.3)
    plt.imshow(false, cmap='Reds', alpha=0.3)
    plt.savefig(f'tmp/{experiment_name}-{architecture}-{encoder.replace("_", "")}-masked-image')
    plt.close()


def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def preprocess(image, lung_mask, lesion_mask, use_transforms):
    image = tf.to_pil_image(image.astype('float32'))
    lesion_mask = tf.to_pil_image(lesion_mask.astype('uint8'))
    lung_mask = tf.to_pil_image(lung_mask.astype('uint8'))
    image = tf.resize(image, [512, 512])
    lesion_mask = tf.resize(lesion_mask, [512, 512])
    lung_mask = tf.resize(lung_mask, [512, 512])
    if use_transforms:
        if np.random.rand() > 0.5:
            image = tf.hflip(image)
            lesion_mask = tf.hflip(lesion_mask)
            lung_mask = tf.hflip(lung_mask)
        if np.random.rand() > 0.5:
            image = tf.vflip(image)
            lesion_mask = tf.vflip(lesion_mask)
            lung_mask = tf.vflip(lung_mask)
        scale = np.random.rand() + 0.5
        rotation = 360 * np.random.rand() - 180
        image = tf.affine(image, rotation, [0, 0], scale, 0)
        lesion_mask = tf.affine(lesion_mask, rotation, [0, 0], scale, 0)
        lung_mask = tf.affine(lung_mask, rotation, [0, 0], scale, 0)
    image = tf.to_tensor(image)
    lesion_mask = tf.to_tensor(lesion_mask)
    lung_mask = tf.to_tensor(lung_mask)
    image = tf.normalize(image, -500, 500)
    lesion_mask = lesion_mask.bool()
    lung_mask = lung_mask.bool()
    return image, lung_mask, lesion_mask


class MedicalSegmentation1(Dataset):
    def __init__(self, index_range, use_transforms):
        urls = ['https://drive.google.com/uc?id=1SJoMelgRqb0EuqlTuq6dxBWf2j9Kno8S', 'https://drive.google.com/uc?id=1MEqpbpwXjrLrH42DqDygWeSkDq0bi92f', 'https://drive.google.com/uc?id=1zj4N_KV0LBko1VSQ7FPZ38eaEGNU0K6-']
        files = ['tr_im.nii.gz', 'tr_mask.nii.gz', 'tr_lungmasks_updated.nii.gz']
        for url, file in zip(urls, files):
            if not os.path.isfile(f'tmp/{file}'):
                gdown.download(url, f'tmp/{file}', quiet=False)
        imgs_path = 'tmp/tr_im.nii.gz'
        images = nib.load(imgs_path)
        self.images = images.get_fdata()[..., index_range]
        lesion_masks_path = 'tmp/tr_mask.nii.gz'
        lesion_masks = nib.load(lesion_masks_path)
        self.lesion_masks = lesion_masks.get_fdata()[..., index_range]
        lung_masks_path = 'tmp/tr_lungmasks_updated.nii.gz'
        lung_masks = nib.load(lung_masks_path)
        self.lung_masks = lung_masks.get_fdata()[..., index_range]
        self.use_transforms = use_transforms

    def __getitem__(self, index):
        image, lung_mask, lesion_mask = preprocess(self.images[..., index], self.lung_masks[..., index], self.lesion_masks[..., index], self.use_transforms)
        return image, lung_mask, lesion_mask

    def __len__(self):
        return self.images.shape[-1]


class MedicalSegmentation2(Dataset):
    def __init__(self, index_volume, use_transforms):
        urls = ['https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m', 'https://drive.google.com/uc?id=1gVuDwFeAGa6jIVX9MeJV5ByIHFpOo5Bp', 'https://drive.google.com/uc?id=1MIp89YhuAKh4as2v_5DUoExgt6-y3AnH']
        files = ['rp_im.zip', 'rp_msk.zip', 'rp_lung_msk.zip']
        for url, file in zip(urls, files):
            zip_path = f'tmp/{file}'
            if not os.path.isfile(zip_path):
                gdown.download(url, zip_path, quiet=False)
                with ZipFile(zip_path, 'r') as z:
                    z.extractall('tmp')
        images_fullname_list = glob.glob('tmp/rp_im/*.nii.gz')
        images_fullname_list.sort()
        images = nib.load(images_fullname_list[index_volume])
        self.images = images.get_fdata()
        lesion_masks_fullname_list = glob.glob('tmp/rp_msk/*.nii.gz')
        lesion_masks_fullname_list.sort()
        lesion_masks = nib.load(lesion_masks_fullname_list[index_volume])
        self.lesion_masks = lesion_masks.get_fdata()
        lung_masks_fullname_list = glob.glob('tmp/rp_lung_msk/*.nii.gz')
        lung_masks_fullname_list.sort()
        lung_masks = nib.load(lung_masks_fullname_list[index_volume])
        self.lung_masks = lung_masks.get_fdata()
        self.use_transforms = use_transforms

    def __getitem__(self, index):
        image, lung_mask, lesion_mask = preprocess(self.images[..., index], self.lung_masks[..., index], self.lesion_masks[..., index], self.use_transforms)
        return image, lung_mask, lesion_mask

    def __len__(self):
        return self.images.shape[-1]


class CTSegBenchmark(Dataset):
    def __init__(self, index_range, use_transforms):
        urls = ['https://zenodo.org/record/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1', 'https://zenodo.org/record/3757476/files/Infection_Mask.zip?download=1', 'https://zenodo.org/record/3757476/files/Lung_Mask.zip?download=1']
        files = ['COVID-19-CT-Seg_20cases', 'Infection_Mask', 'Lung_Mask']
        for url, file in zip(urls, files):
            zip_file = f'tmp/{file}.zip'
            if not os.path.isfile(zip_file):
                r = requests.get(url)
                with open(zip_file, 'wb') as f:
                    f.write(r.content)
                with ZipFile(zip_file, 'r') as z:
                    z.extractall(f'tmp/{file}/')
        images = np.array([]).reshape(512, 512, 0)
        for fullname in glob.glob(f'tmp/{files[0]}/*.nii.gz'):
            images_ = nib.load(fullname)
            images_ = np.resize(images_.get_fdata(), (512, 512, images_.shape[-1]))
            images = np.concatenate((images, images_), 2)
        self.images = images[..., index_range]
        lesion_masks = np.array([]).reshape(512, 512, 0)
        for fullname in glob.glob(f'tmp/{files[1]}/*.nii.gz'):
            lesion_masks_ = nib.load(fullname)
            lesion_masks_ = np.resize(lesion_masks_.get_fdata(), (512, 512, lesion_masks_.shape[-1]))
            lesion_masks = np.concatenate((lesion_masks, lesion_masks_), 2)
        self.lesion_masks = lesion_masks[..., index_range]
        lung_masks = np.array([]).reshape(512, 512, 0)
        for fullname in glob.glob(f'tmp/{files[2]}/*.nii.gz'):
            lung_masks_ = nib.load(fullname)
            lung_masks_ = np.resize(lung_masks_.get_fdata(), (512, 512, lung_masks.shape[-1]))
            lung_masks = np.concatenate((lung_masks, lung_masks_), 2)
        self.lung_masks = lung_masks[..., index_range]
        self.use_transforms = use_transforms

    def __getitem__(self, index):
        image, lung_mask, lesion_mask = preprocess(self.images[..., index], self.lung_masks[..., index], self.lesion_masks[..., index], self.use_transforms)
        return image, lung_mask, lesion_mask

    def __len__(self):
        return self.images.shape[-1]


if __name__ == '__main__':
    # Set appropriate variables (e.g. num_samples) to a lower value to reduce the computational cost of the draft (fast) version document.
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, action='store_true')
    args = parser.parse_args()
    if args.full:
        num_epochs = 100
        index_train_range = range(80)
        index_validation_range = range(80, 100)
        index_test_volume_range = range(9)
        encoder_list = ['vgg11', 'vgg13', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnext50_32x4d', 'dpn68', 'dpn98', 'mobilenet_v2', 'xception', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6']
    else:
        num_epochs = 2
        index_train_range = range(1)
        index_validation_range = range(2, 4)
        index_test_volume_range = range(1)
        encoder_list = ['vgg11', 'resnet18', 'mobilenet_v2', 'efficientnet-b0']

    # Set random seeds for reproducibility.
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    architecture_list = [Unet, Linknet, FPN, PSPNet]
    architecture_name_list = [architecture.__name__ for architecture in architecture_list]
    experiment_list = ['Lung segmentation', 'Lesion segmentation A', 'Lesion segmentation B']
    experiment_name_list = [experiment.lower().replace(' ', '-') for experiment in experiment_list]
    encoder_weights_list = [None, 'imagenet']
    train_dataset = MedicalSegmentation1(index_train_range, use_transforms=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = MedicalSegmentation1(index_validation_range, use_transforms=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    dice_loss = DiceLoss()
    metric_list = ['Sens', 'Spec', 'Dice']
    metrics_array = np.zeros((len(experiment_list), len(architecture_list), len(encoder_list), len(encoder_weights_list), len(metric_list)))
    num_parameters_array = np.zeros((len(architecture_list), len(encoder_list)))
    hist_bins = 100
    hist_range = [-2, 2]
    hist_images_array = np.zeros((len(experiment_list), hist_bins))
    hist_masks_array = np.zeros_like(hist_images_array)
    train_loss_array = np.zeros((len(experiment_list), len(architecture_list), len(encoder_list), len(encoder_weights_list), num_epochs))
    validation_loss_array = np.zeros_like(train_loss_array)
    train_time_array = np.zeros((len(experiment_list), len(architecture_list), len(encoder_list), len(encoder_weights_list)))
    validation_time_array = np.zeros_like(train_time_array)
    for index_experiment_name, experiment_name in enumerate(experiment_name_list):
        for index_architecture, (architecture, architecture_name) in enumerate(zip(architecture_list, architecture_name_list)):
            for index_encoder, encoder in enumerate(encoder_list):
                for index_encoder_weights, encoder_weights in enumerate(encoder_weights_list):
                    model = architecture(encoder, encoder_weights=encoder_weights, activation='sigmoid', in_channels=1).to(device)
                    num_parameters_array[index_architecture, index_encoder] = get_num_parameters(model)
                    optimizer = optim.Adam(model.parameters())
                    validation_loss_best = float('inf')
                    model_path = f'tmp/{experiment_name}-{architecture_name}-{encoder}-{encoder_weights}.pt'
                    for index_epoch, epoch in enumerate(range(num_epochs)):
                        train_loss_sum = 0
                        model.train()
                        for images, lung_masks, lesion_masks in train_dataloader:
                            if experiment_name == experiment_name_list[0]:
                                masks = lung_masks
                            elif experiment_name == experiment_name_list[1]:
                                masks = lesion_masks
                            elif experiment_name == experiment_name_list[2]:
                                images *= lung_masks
                                masks = lesion_masks
                            images = images.to(device)
                            masks = masks.to(device)
                            optimizer.zero_grad()
                            if device == 'cuda':
                                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                                    predictions = model(images)
                                train_time_array[index_experiment_name, index_architecture, index_encoder, index_encoder_weights] += sum([item.cuda_time for item in prof.function_events])
                            else:
                                with torch.autograd.profiler.profile(use_cuda=False) as prof:
                                    predictions = model(images)
                                train_time_array[index_experiment_name, index_architecture, index_encoder, index_encoder_weights] += sum([item.cpu_time for item in prof.function_events])
                            if (architecture_name == architecture_name_list[0]) and (encoder == encoder_list[0]) and (encoder_weights == encoder_weights_list[0]) and (index_epoch == num_epochs - 1):
                                save_image(images[0, 0], experiment_name)
                                save_masked_image(images[0, 0], masks[0, 0], masks[0, 0], experiment_name, 'mask', 'train')
                                save_masked_image(images[0, 0], masks[0, 0], predictions[0, 0], experiment_name, 'prediction', 'train')
                            if (architecture_name == architecture_name_list[0]) and (encoder == 'resnet18'):
                                if index_epoch == 0:
                                    save_weights(model, experiment_name, architecture_name, f'{encoder_weights}-before')
                                elif index_epoch == num_epochs - 1:
                                    save_weights(model, experiment_name, architecture_name, f'{encoder_weights}-after')
                            loss = dice_loss(predictions, masks)
                            loss.backward()
                            optimizer.step()
                            train_loss_sum += loss.item()
                        train_loss = train_loss_sum / len(train_dataloader)
                        train_loss_array[index_experiment_name, index_architecture, index_encoder, index_encoder_weights, index_epoch] = train_loss
                        validation_loss_sum = 0
                        model.eval()
                        with torch.no_grad():
                            for images, lung_masks, lesion_masks in validation_dataloader:
                                if experiment_name == experiment_name_list[0]:
                                    masks = lung_masks
                                elif experiment_name == experiment_name_list[1]:
                                    masks = lesion_masks
                                elif experiment_name == experiment_name_list[2]:
                                    images *= lung_masks
                                    masks = lesion_masks
                                images = images.to(device)
                                masks = masks.to(device)
                                if device == 'cuda':
                                    with torch.autograd.profiler.profile(use_cuda=True) as prof:
                                        predictions = model(images)
                                    validation_time_array[index_experiment_name, index_architecture, index_encoder, index_encoder_weights] += sum([item.cuda_time for item in prof.function_events])
                                else:
                                    with torch.autograd.profiler.profile(use_cuda=False) as prof:
                                        predictions = model(images)
                                    validation_time_array[index_experiment_name, index_architecture, index_encoder, index_encoder_weights] += sum([item.cpu_time for item in prof.function_events])
                                loss = dice_loss(predictions, masks)
                                validation_loss_sum += loss.item()
                            validation_loss = validation_loss_sum / len(validation_dataloader)
                            validation_loss_array[index_experiment_name, index_architecture, index_encoder, index_encoder_weights, index_epoch] = validation_loss
                            print(f'{experiment_name}, {architecture_name}, {encoder}, {encoder_weights}, epoch: {epoch}, train loss: {train_loss:.3f}, validation loss: {validation_loss:.3f}')
                            if validation_loss < validation_loss_best:
                                validation_loss_best = validation_loss
                                torch.save(model.state_dict(), model_path)
                                print('saving as best model')
                    model = architecture(encoder, encoder_weights=encoder_weights, activation='sigmoid', in_channels=1).to(device)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    num_slices_test = 0
                    for index_test_volume in index_test_volume_range:
                        test_dataset = MedicalSegmentation2(index_test_volume, use_transforms=False)
                        test_dataloader = DataLoader(test_dataset)
                        num_slices_test += len(test_dataloader)
                        with torch.no_grad():
                            for images, lung_masks, lesion_masks in test_dataloader:
                                if experiment_name == experiment_name_list[0]:
                                    masks = lung_masks
                                elif experiment_name == experiment_name_list[1]:
                                    masks = lesion_masks
                                elif experiment_name == experiment_name_list[2]:
                                    images *= lung_masks
                                    masks = lesion_masks
                                images = images.to(device)
                                masks = masks.to(device)
                                predictions = model(images)
                                if (architecture_name == architecture_name_list[0]) and (encoder == encoder_list[0]) and (encoder_weights == encoder_weights_list[0]):
                                    hist_images_array[index_experiment_name] += np.histogram(images.cpu(), hist_bins, hist_range)[0]
                                    hist_masks_array[index_experiment_name] += np.histogram(images.cpu() * masks.cpu(), hist_bins, hist_range)[0]
                                metrics_values = metrics(predictions, masks)
                                metrics_array[index_experiment_name, index_architecture, index_encoder, index_encoder_weights] += metrics_values
                            if index_test_volume == 0:
                                index_test = 12
                                images = test_dataset[index_test][0]
                                lung_masks = test_dataset[index_test][1]
                                lesion_masks = test_dataset[index_test][2]
                                if experiment_name == experiment_name_list[0]:
                                    masks = lung_masks
                                elif experiment_name == experiment_name_list[1]:
                                    masks = lesion_masks
                                elif experiment_name == experiment_name_list[2]:
                                    images *= lung_masks
                                    masks = lesion_masks
                                predictions = model(images.unsqueeze(0).to(device))
                                save_masked_image(images[0], masks[0], predictions[0, 0], experiment_name, architecture_name, encoder)
                            if (index_test_volume == 0) and (encoder == 'resnet18') and (encoder_weights is None):
                                volume_mask = np.zeros((512, 512, len(test_dataset)))
                                volume_prediction = np.zeros((512, 512, len(test_dataset)))
                                for index_slice_volume, (images, lung_masks, lesion_masks) in enumerate(test_dataset):
                                    if experiment_name == experiment_name_list[0]:
                                        masks = lung_masks
                                    elif experiment_name == experiment_name_list[1]:
                                        masks = lesion_masks
                                    elif experiment_name == experiment_name_list[2]:
                                        images *= lung_masks
                                        masks = lesion_masks
                                    volume_mask[:, :, index_slice_volume] = masks[0].float()
                                    predictions = model(images.unsqueeze(0).to(device))
                                    volume_prediction[:, :, index_slice_volume] = predictions[0].float().cpu()
                                volume_mask = volume_mask[:, :, ::-1]
                                if architecture_name == 'Unet':
                                    save_3d(volume_mask, args.full, experiment_name, 'mask', '')
                                volume_prediction = volume_prediction[:, :, ::-1]
                                save_3d(volume_prediction, args.full, experiment_name, architecture_name, encoder_weights)
                    if not args.full:
                        os.remove(model_path)

    for hist_images, hist_masks, experiment_name in zip(hist_images_array, hist_masks_array, experiment_name_list):
        save_hist(hist_images, hist_masks, hist_range, experiment_name)

    for experiment_name, train_loss, validation_loss in zip(experiment_name_list, train_loss_array, validation_loss_array):
        save_loss(train_loss, 'Train', experiment_name, architecture_name_list, [0, 1])
        save_loss(validation_loss, 'Validation', experiment_name, architecture_name_list, [0, 1])
    save_loss(train_loss_array[2] - train_loss_array[1], 'Train diff', experiment_name, architecture_name_list, [-0.4, 0.4])
    save_loss(validation_loss_array[2] - validation_loss_array[1], 'Validation diff', experiment_name, architecture_name_list, [-0.4, 0.4])

    metrics_array = 100 * metrics_array / num_slices_test
    num_parameters_array = num_parameters_array / 10 ** 6
    for experiment, experiment_name, metrics_ in zip(experiment_list, experiment_name_list, metrics_array):
        save_architecture_box(metrics_[..., -1], architecture_name_list, experiment_name)
        save_scatter(num_parameters_array, metrics_[..., -1], architecture_name_list, experiment_name, [70, 100])
    save_scatter(num_parameters_array, metrics_array[2, ..., -1] - metrics_array[1, ..., -1], architecture_name_list, 'diff', [-15, 15])
    save_initialization_box(metrics_array[..., -1], encoder_weights_list)

    encoder_weights_mean = metrics_array[..., -1].reshape(-1, len(encoder_weights_list)).mean(0).round(2)
    encoder_weights_std = metrics_array[..., -1].reshape(-1, len(encoder_weights_list)).std(0).round(2)

    encoder_mean = metrics_array.transpose([2, 0, 1, 3, 4]).reshape(metrics_array.shape[2], -1).mean(1)
    metrics_array = metrics_array.transpose([1, 2, 3, 0, 4])
    metrics_array_global_mean = metrics_array.reshape(-1, np.prod(metrics_array.shape[2:])).mean(0)
    metrics_array_global_std = metrics_array.reshape(-1, np.prod(metrics_array.shape[2:])).std(0)
    metrics_array = np.concatenate((metrics_array, metrics_array.mean(1, keepdims=True), metrics_array.std(1, keepdims=True)), 1)
    metrics_array = metrics_array.reshape(-1, np.prod(metrics_array.shape[2:]))

    num_parameters_array_global_mean = num_parameters_array.mean()
    num_parameters_array_global_std = num_parameters_array.std()
    num_parameters_array = np.concatenate((num_parameters_array, num_parameters_array.mean(1, keepdims=True), num_parameters_array.std(1, keepdims=True)), 1)

    train_time_array /= num_epochs * 10 ** 6
    train_time_array = train_time_array.mean(0).mean(-1)
    train_time_array_global_mean = train_time_array.mean()
    train_time_array_global_std = train_time_array.std()
    train_time_array = np.concatenate((train_time_array, train_time_array.mean(1, keepdims=True), train_time_array.std(1, keepdims=True)), 1)

    validation_time_array /= num_epochs * 10 ** 6
    validation_time_array = validation_time_array.mean(0).mean(-1)
    validation_time_array_global_mean = validation_time_array.mean()
    validation_time_array_global_std = validation_time_array.std()
    validation_time_array = np.concatenate((validation_time_array, validation_time_array.mean(1, keepdims=True), validation_time_array.std(1, keepdims=True)), 1)

    index = pd.MultiIndex.from_product([architecture_name_list, [e.replace('_', '') for e in encoder_list] + ['Mean', 'Std']])
    multicolumn = pd.MultiIndex.from_product([[str(e) for e in encoder_weights_list], experiment_list, metric_list])
    df = pd.DataFrame(metrics_array, index=index, columns=multicolumn)
    df['Performance', 'related', 'Pars(M)'] = num_parameters_array.flatten()
    df['Performance', 'related', 'Train(s)'] = train_time_array.flatten()
    df['Performance', 'related', 'Val(s)'] = validation_time_array.flatten()
    df.loc[('Global', 'Mean'), :] = np.append(metrics_array_global_mean, (num_parameters_array_global_mean, train_time_array_global_mean, validation_time_array_global_mean))
    df.loc[('Global', 'Std'), :] = np.append(metrics_array_global_std, (num_parameters_array_global_std, train_time_array_global_std, validation_time_array_global_std))
    df.index.names = ['Architecture', 'Encoder']
    df = df.round(2)
    max_per_column_list = df.max(0)
    min_per_column_list = df.min(0)
    index_max_per_column_list = df.idxmax(0)
    index_min_per_column_list = df.idxmin(0)
    formatters = [lambda x, max_per_column=max_per_column: fr'\textbf{{{x:.2f}}}' if (x == max_per_column) else f'{x:.2f}' for max_per_column in max_per_column_list]
    formatters[-3:] = [lambda x, min_per_column=min_per_column: fr'\textbf{{{x:.2f}}}' if (x == min_per_column) else f'{x:.2f}' for min_per_column in min_per_column_list[-3:]]
    df.to_latex('tmp/metrics.tex', formatters=formatters, bold_rows=True, multirow=True, multicolumn=True, multicolumn_format='c', escape=False)

    df_keys_values = pd.DataFrame(
        {
            'key': ['num-epochs', 'batch-size', 'num-slices-test', 'encoder-best', 'encoder-worst']
            + [f'{experiment_name}-{encoder_weights}-mean' for experiment_name, encoder_weights in itertools.product(experiment_name_list, encoder_weights_list)]
            + [f'{experiment_name}-{encoder_weights}-std' for experiment_name, encoder_weights in itertools.product(experiment_name_list, encoder_weights_list)]
            + [f'{experiment_name}-{encoder_weights}-max' for experiment_name, encoder_weights in itertools.product(experiment_name_list, encoder_weights_list)]
            + [f'{encoder_weights}-{stat}' for encoder_weights, stat in itertools.product(encoder_weights_list, ['mean', 'std'])]
            + [f'{experiment_name}-architecture-{encoder_weights}-index-max' for experiment_name, encoder_weights in itertools.product(experiment_name_list, encoder_weights_list)]
            + [f'{experiment_name}-encoder-{encoder_weights}-index-max' for experiment_name, encoder_weights in itertools.product(experiment_name_list, encoder_weights_list)]
            + [f'{experiment_name}-{architecture_name}-{encoder_weights}-mean' for experiment_name, architecture_name, encoder_weights in itertools.product(experiment_name_list, architecture_name_list, encoder_weights_list)]
            + [f'{experiment_name}-{architecture_name}-{encoder_weights}-std' for experiment_name, architecture_name, encoder_weights in itertools.product(experiment_name_list, architecture_name_list, encoder_weights_list)],
            'value': [str(int(num_epochs)), str(int(batch_size)), str(int(num_slices_test)), encoder_list[encoder_mean.argmax()].replace('_', ''), encoder_list[encoder_mean.argmin()].replace('_', '')]
            + [df.loc['Global', 'Mean'][str(encoder_weights), experiment]['Dice'] for experiment, encoder_weights in itertools.product(experiment_list, encoder_weights_list)]
            + [df.loc['Global', 'Std'][str(encoder_weights), experiment]['Dice'] for experiment, encoder_weights in itertools.product(experiment_list, encoder_weights_list)]
            + [max_per_column_list[2], max_per_column_list[11], max_per_column_list[5], max_per_column_list[14], max_per_column_list[8], max_per_column_list[17], encoder_weights_mean[0], encoder_weights_mean[1], encoder_weights_std[0], encoder_weights_std[1], index_max_per_column_list[2][0], index_max_per_column_list[2][1], index_max_per_column_list[11][0], index_max_per_column_list[11][1], index_max_per_column_list[5][0], index_max_per_column_list[5][1], index_max_per_column_list[14][0], index_max_per_column_list[14][1], index_max_per_column_list[8][0], index_max_per_column_list[8][1], index_max_per_column_list[17][0], index_max_per_column_list[17][1]]
            + [df.loc[architecture_name, 'Mean'][str(encoder_weights), experiment]['Dice'] for experiment, architecture_name, encoder_weights in itertools.product(experiment_list, architecture_name_list, encoder_weights_list)]
            + [df.loc[architecture_name, 'Std'][str(encoder_weights), experiment]['Dice'] for experiment, architecture_name, encoder_weights in itertools.product(experiment_list, architecture_name_list, encoder_weights_list)],
        }
    )
    df_keys_values.to_csv('tmp/keys-values.csv')
