import gdown
import glob
import nibabel as nib
import numpy as np
import os
import requests
import torch

from torch.utils.data import Dataset
from zipfile import ZipFile
from torchvision.transforms import functional as tf

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
    def __init__(self, cache_dir, index_range, use_transforms):
        urls = ['https://drive.google.com/uc?id=1SJoMelgRqb0EuqlTuq6dxBWf2j9Kno8S', 'https://drive.google.com/uc?id=1MEqpbpwXjrLrH42DqDygWeSkDq0bi92f', 'https://drive.google.com/uc?id=1zj4N_KV0LBko1VSQ7FPZ38eaEGNU0K6-']
        files = ['tr_im.nii.gz', 'tr_mask.nii.gz', 'tr_lungmasks_updated.nii.gz']
        for url, file in zip(urls, files):
            if not os.path.isfile(f'{cache_dir}/{file}'):
                gdown.download(url, f'{cache_dir}/{file}', quiet=False)
        imgs_path = f'{cache_dir}/tr_im.nii.gz'
        images = nib.load(imgs_path)
        self.images = images.get_fdata()[..., index_range]
        lesion_masks_path = f'{cache_dir}/tr_mask.nii.gz'
        lesion_masks = nib.load(lesion_masks_path)
        self.lesion_masks = lesion_masks.get_fdata()[..., index_range]
        lung_masks_path = f'{cache_dir}/tr_lungmasks_updated.nii.gz'
        lung_masks = nib.load(lung_masks_path)
        self.lung_masks = lung_masks.get_fdata()[..., index_range]
        self.use_transforms = use_transforms

    def __getitem__(self, index):
        image, lung_mask, lesion_mask = preprocess(self.images[..., index], self.lung_masks[..., index], self.lesion_masks[..., index], self.use_transforms)
        return image, lung_mask, lesion_mask

    def __len__(self):
        return self.images.shape[-1]


class MedicalSegmentation2(Dataset):
    def __init__(self, cache_dir, index_volume, use_transforms):
        urls = ['https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m', 'https://drive.google.com/uc?id=1gVuDwFeAGa6jIVX9MeJV5ByIHFpOo5Bp', 'https://drive.google.com/uc?id=1MIp89YhuAKh4as2v_5DUoExgt6-y3AnH']
        files = ['rp_im.zip', 'rp_msk.zip', 'rp_lung_msk.zip']
        for url, file in zip(urls, files):
            zip_path = f'{cache_dir}/{file}'
            if not os.path.isfile(zip_path):
                gdown.download(url, zip_path, quiet=False)
                with ZipFile(zip_path, 'r') as z:
                    z.extractall(f'{cache_dir}')

        images_fullname_list = glob.glob(f'{cache_dir}/rp_im/*.nii.gz')
        images_fullname_list.sort()
        images = nib.load(images_fullname_list[index_volume])
        self.images = images.get_fdata()

        lesion_masks_fullname_list = glob.glob(f'{cache_dir}/rp_msk/*.nii.gz')
        lesion_masks_fullname_list.sort()
        lesion_masks = nib.load(lesion_masks_fullname_list[index_volume])
        self.lesion_masks = lesion_masks.get_fdata()

        lung_masks_fullname_list = glob.glob(f'{cache_dir}/rp_lung_msk/*.nii.gz')
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
    def __init__(self, cache_dir, index_volume, index_range, use_transforms):
        urls = ['https://zenodo.org/record/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1', 'https://zenodo.org/record/3757476/files/Infection_Mask.zip?download=1', 'https://zenodo.org/record/3757476/files/Lung_Mask.zip?download=1']
        files = ['COVID-19-CT-Seg_20cases', 'Infection_Mask', 'Lung_Mask']
        for url, file in zip(urls, files):
            zip_file = f'{cache_dir}/{file}.zip'
            if not os.path.isfile(zip_file):
                r = requests.get(url)
                with open(zip_file, 'wb') as f:
                    f.write(r.content)
                with ZipFile(zip_file, 'r') as z:
                    z.extractall(f'{cache_dir}/{file}/')

        images = np.array([]).reshape(512, 512, 0)
        for fullname in glob.glob(f'{cache_dir}/{files[0]}/*.nii.gz'):
            images_ = nib.load(fullname)
            images_ = np.resize(images_.get_fdata(), (512, 512, images_.shape[-1]))
            images = np.concatenate((images, images_), 2)
        self.images = images[..., index_range]
        lesion_masks = np.array([]).reshape(512, 512, 0)
        for fullname in glob.glob(f'{cache_dir}/{files[1]}/*.nii.gz'):
            lesion_masks_ = nib.load(fullname)
            lesion_masks_ = np.resize(lesion_masks_.get_fdata(), (512, 512, lesion_masks_.shape[-1]))
            lesion_masks = np.concatenate((lesion_masks, lesion_masks_), 2)
        self.lesion_masks = lesion_masks[..., index_range]
        lung_masks = np.array([]).reshape(512, 512, 0)
        for fullname in glob.glob(f'{cache_dir}/{files[2]}/*.nii.gz'):
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
