import glob
import itertools
import ssl
from os import environ
from pathlib import Path
from shutil import move, rmtree
from zipfile import ZipFile

import gdown
import nibabel as nib
import numpy as np
import onnx
import pandas as pd
import requests
import torch
from fvcore.nn import FlopCountAnalysis
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from onnx_tf.backend import prepare
from scipy.stats import gaussian_kde
from segmentation_models_pytorch import FPN, Linknet, PSPNet, Unet, metrics
from segmentation_models_pytorch.utils.losses import DiceLoss
from skimage.measure import marching_cubes
from tensorflowjs.converters import tf_saved_model_conversion_v2
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tf


class CTSegBenchmark(Dataset): # type: ignore[type-arg]

    def __getitem__(self: "CTSegBenchmark", index: int) -> tuple: # type: ignore[type-arg]
        image, mask_lung, mask_lesion = preprocess_image(self.images[..., index], self.mask_lesions[..., index], self.mask_lungs[..., index], self.use_transforms, self.rng)
        return (image, mask_lung, mask_lesion)

    def __init__(self: "CTSegBenchmark", index_range: range, use_transforms: bool) -> None: # noqa: FBT001
        self.rng = np.random.default_rng(seed=0)
        urls = ["https://zenodo.org/record/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1", "https://zenodo.org/record/3757476/files/Infection_Mask.zip?download=1", "https://zenodo.org/record/3757476/files/Lung_Mask.zip?download=1"]
        file_names = ["COVID-19-CT-Seg_20cases", "Infection_Mask", "Lung_Mask"]
        for url, file_name in zip(urls, file_names, strict=True):
            zip_file_path = Path("data") / f"{file_name}.zip"
            if not zip_file_path.is_file():
                response = requests.get(url, timeout=60)
                with zip_file_path.open("wb") as file:
                    file.write(response.content)
                with ZipFile(zip_file_path, "r") as zip_file:
                    zip_file.extractall(f"data/{file_name}")
        images: np.ndarray = np.array([]).reshape(512, 512, 0) # type: ignore[type-arg]
        for file_path in glob.glob(f"data/{file_names[0]}/*.nii.gz"):
            images_ = nib.load(file_path) # type: ignore[attr-defined]
            images_ = np.resize(images_.get_fdata(), (512, 512, images_.shape[-1]))
            images = np.concatenate((images, images_), 2)
        self.images = images[..., index_range]
        mask_lesions: np.ndarray = np.array([]).reshape(512, 512, 0) # type: ignore[type-arg]
        for file_path in glob.glob(f"data/{file_names[1]}/*.nii.gz"):
            mask_lesions_ = nib.load(file_path) # type: ignore[attr-defined]
            mask_lesions_ = np.resize(mask_lesions_.get_fdata(), (512, 512, mask_lesions_.shape[-1]))
            mask_lesions = np.concatenate((mask_lesions, mask_lesions_), 2)
        self.mask_lesions = mask_lesions[..., index_range]
        mask_lungs: np.ndarray = np.array([]).reshape(512, 512, 0) # type: ignore[type-arg]
        for file_path in glob.glob(f"data/{file_names[2]}/*.nii.gz"):
            mask_lungs_ = nib.load(file_path) # type: ignore[attr-defined]
            mask_lungs_ = np.resize(mask_lungs_.get_fdata(), (512, 512, mask_lungs.shape[-1]))
            mask_lungs = np.concatenate((mask_lungs, mask_lungs_), 2)
        self.mask_lungs = mask_lungs[..., index_range]
        self.use_transforms = use_transforms

    def __len__(self: "CTSegBenchmark") -> int:
        return self.images.shape[-1]


class MedicalSegmentation1(Dataset): # type: ignore[type-arg]

    def __getitem__(self: "MedicalSegmentation1", index: int) -> tuple: # type: ignore[type-arg]
        image, mask_lung, mask_lesion = preprocess_image(self.images[..., index], self.mask_lesions[..., index], self.mask_lungs[..., index], self.use_transforms, self.rng)
        return (image, mask_lung, mask_lesion)

    def __init__(self: "MedicalSegmentation1", index_range: range, use_transforms: bool) -> None: # noqa: FBT001
        self.rng = np.random.default_rng(seed=0)
        urls = ["https://drive.google.com/uc?id=1SJoMelgRqb0EuqlTuq6dxBWf2j9Kno8S", "https://drive.google.com/uc?id=1MEqpbpwXjrLrH42DqDygWeSkDq0bi92f", "https://drive.google.com/uc?id=1zj4N_KV0LBko1VSQ7FPZ38eaEGNU0K6-"]
        file_names = ["tr_im.nii.gz", "tr_mask.nii.gz", "tr_lungmasks_updated.nii.gz"]
        for url, file_name in zip(urls, file_names, strict=True):
            nifti_file_path = Path("data") / file_name
            if not nifti_file_path.is_file():
                gdown.download(url, nifti_file_path.as_posix(), quiet=False)
        images = nib.load("data/tr_im.nii.gz") # type: ignore[attr-defined]
        self.images = images.get_fdata()[..., index_range]
        mask_lesions = nib.load("data/tr_mask.nii.gz") # type: ignore[attr-defined]
        self.mask_lesions = mask_lesions.get_fdata()[..., index_range]
        mask_lungs = nib.load("data/tr_lungmasks_updated.nii.gz") # type: ignore[attr-defined]
        self.mask_lungs = mask_lungs.get_fdata()[..., index_range]
        self.use_transforms = use_transforms

    def __len__(self: "MedicalSegmentation1") -> int:
        output: int = self.images.shape[-1]
        return output


class MedicalSegmentation2(Dataset): # type: ignore[type-arg]

    def __getitem__(self: "MedicalSegmentation2", index: int) -> tuple: # type: ignore[type-arg]
        image, mask_lung, mask_lesion = preprocess_image(self.images[..., index], self.mask_lesions[..., index], self.mask_lungs[..., index], self.use_transforms, self.rng)
        return (image, mask_lung, mask_lesion)

    def __init__(self: "MedicalSegmentation2", index_volume: int, use_transforms: bool) -> None: # noqa: FBT001
        self.rng = np.random.default_rng(seed=0)
        urls = ["https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m", "https://drive.google.com/uc?id=1gVuDwFeAGa6jIVX9MeJV5ByIHFpOo5Bp", "https://drive.google.com/uc?id=1MIp89YhuAKh4as2v_5DUoExgt6-y3AnH"]
        file_names = ["rp_im.zip", "rp_msk.zip", "rp_lung_msk.zip"]
        for url, file_name in zip(urls, file_names, strict=True):
            zip_file_path = Path("data") / file_name
            if not zip_file_path.is_file():
                gdown.download(url, zip_file_path.as_posix(), quiet=False)
                with ZipFile(zip_file_path, "r") as zip_file:
                    zip_file.extractall("data")
        image_file_paths = sorted(glob.glob("data/rp_im/*.nii.gz"))
        images = nib.load(image_file_paths[index_volume]) # type: ignore[attr-defined]
        self.images = images.get_fdata()
        mask_lesions_file_paths = sorted(glob.glob("data/rp_msk/*.nii.gz"))
        mask_lesions = nib.load(mask_lesions_file_paths[index_volume]) # type: ignore[attr-defined]
        self.mask_lesions = mask_lesions.get_fdata()
        mask_lungs_file_paths = sorted(glob.glob("data/rp_lung_msk/*.nii.gz"))
        mask_lungs = nib.load(mask_lungs_file_paths[index_volume]) # type: ignore[attr-defined]
        self.mask_lungs = mask_lungs.get_fdata()
        self.use_transforms = use_transforms

    def __len__(self: "MedicalSegmentation2") -> int:
        output: int = self.images.shape[-1]
        return output


def calculate_metrics(mask: torch.Tensor, prediction: torch.Tensor) -> tuple[float, float, float]:
    tp, fp, fn, tn = metrics.get_stats(prediction, mask, mode="binary", threshold=0.5)
    f1_score = metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    sensitivity = metrics.sensitivity(tp, fp, fn, tn)
    specificity = metrics.specificity(tp, fp, fn, tn)
    return (sensitivity.item(), specificity.item(), f1_score.item())


def main() -> None: # noqa: C901, PLR0912, PLR0915
    bin_file_path = Path("bin")
    if not bin_file_path.exists():
        bin_file_path.mkdir(parents=True)
    data_file_path = Path("data")
    if not data_file_path.exists():
        data_file_path.mkdir(parents=True)
    ssl._create_default_https_context = ssl._create_unverified_context # noqa: SLF001
    plt.rcParams["image.interpolation"] = "none"
    plt.rcParams["savefig.format"] = "pdf"
    plt.rcParams["savefig.bbox"] = "tight"
    encoder_names = ["vgg11", "vgg13", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "densenet121", "densenet161", "densenet169", "densenet201", "resnext50_32x4d", "dpn68", "dpn98", "mobilenet_v2", "xception", "inceptionv4", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", "efficientnet-b5", "efficientnet-b6"]
    epochs_num = 100
    range_test_volume = range(9)
    range_train = range(80)
    range_validation = range(80, 100)
    step_size = 1
    if environ["DEBUG"] == "1":
        encoder_names = ["vgg11", "resnet18", "mobilenet_v2", "efficientnet-b0"]
        epochs_num = 2
        range_test_volume = range(1)
        range_train = range(1)
        range_validation = range(2, 4)
        step_size = 10
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    architectures = [Unet, Linknet, FPN, PSPNet]
    architecture_names = [architecture.__name__ for architecture in architectures]
    experiments = ["Lung segmentation", "Lesion segmentation A", "Lesion segmentation B"]
    experiment_names = [experiment.lower().replace(" ", "-") for experiment in experiments]
    encoders_weights = [None, "imagenet"]
    dataset_train = MedicalSegmentation1(range_train, use_transforms=True)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_validation = MedicalSegmentation1(range_validation, use_transforms=False)
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size)
    dice_loss = DiceLoss()
    metric_names = ["Sens", "Spec", "Dice"]
    metrics_array = np.zeros((len(experiments), len(architectures), len(encoder_names), len(encoders_weights), len(metric_names)))
    parameters_num_array = np.zeros((len(architectures), len(encoder_names)))
    hist_bins = 100
    hist_range = (-0.5, 0.5)
    hist_images_array = np.zeros((len(experiments), hist_bins))
    hist_masks_array = np.zeros_like(hist_images_array)
    loss_train_array = np.zeros((len(experiments), len(architectures), len(encoder_names), len(encoders_weights), epochs_num))
    loss_validation_array = np.zeros_like(loss_train_array)
    flops_array = np.zeros((len(experiments), len(architectures), len(encoder_names), len(encoders_weights)))
    for experiment_name_index, experiment_name in enumerate(experiment_names):
        for architecture_index, (architecture, architecture_name) in enumerate(zip(architectures, architecture_names, strict=True)):
            for encoder_name_index, encoder_name in enumerate(encoder_names):
                for encoder_weights_index, encoder_weights in enumerate(encoders_weights):
                    model = architecture(encoder_name, encoder_weights=encoder_weights, activation="sigmoid", in_channels=1).to(device)
                    parameters_num_array[architecture_index, encoder_name_index] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                    optimizer = optim.Adam(model.parameters())
                    loss_validation_best = float("inf")
                    model_file_path = Path("bin") / "{experiment_name}-{architecture_name}-{encoder_name}-{encoder_weights}.pt"
                    flops = FlopCountAnalysis(model, dataloader_train.dataset[0][0].unsqueeze(0).to(device))
                    flops_array[experiment_name_index, architecture_index, encoder_name_index, encoder_weights_index] = flops.total()
                    for epoch_index in range(epochs_num):
                        loss_train_sum = 0
                        model.train()
                        for images, mask_lungs, mask_lesions in dataloader_train:
                            if experiment_name == experiment_names[0]:
                                masks = mask_lungs
                            elif experiment_name == experiment_names[1]:
                                masks = mask_lesions
                            elif experiment_name == experiment_names[2]:
                                images *= mask_lungs # noqa: PLW2901
                                masks = mask_lesions
                            images = images.to(device) # noqa: PLW2901
                            masks = masks.to(device)
                            predictions = model(images)
                            if architecture_name == architecture_names[0] and encoder_name == encoder_names[0] and (encoder_weights == encoders_weights[0]) and (epoch_index == epochs_num - 1):
                                save_figure_image(experiment_name, images[0, 0])
                                save_figure_image_masked("mask", "train", experiment_name, images[0, 0], masks[0, 0], masks[0, 0])
                                save_figure_image_masked("prediction", "train", experiment_name, images[0, 0], masks[0, 0], predictions[0, 0])
                            if architecture_name == architecture_names[0] and encoder_name == "resnet18":
                                if epoch_index == 0:
                                    save_figure_weights(architecture_name, f"{encoder_weights}-before", experiment_name, model)
                                elif epoch_index == epochs_num - 1:
                                    save_figure_weights(architecture_name, f"{encoder_weights}-after", experiment_name, model)
                            loss = dice_loss(predictions, masks)
                            loss_train_sum += loss.item()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        loss_train = loss_train_sum / len(dataloader_train)
                        loss_train_array[experiment_name_index, architecture_index, encoder_name_index, encoder_weights_index, epoch_index] = loss_train
                        loss_validation_sum = 0
                        model.eval()
                        with torch.no_grad():
                            for images, mask_lungs, mask_lesions in dataloader_validation:
                                if experiment_name == experiment_names[0]:
                                    masks = mask_lungs
                                elif experiment_name == experiment_names[1]:
                                    masks = mask_lesions
                                elif experiment_name == experiment_names[2]:
                                    images *= mask_lungs # noqa: PLW2901
                                    masks = mask_lesions
                                images = images.to(device) # noqa: PLW2901
                                masks = masks.to(device)
                                predictions = model(images)
                                loss = dice_loss(predictions, masks)
                                loss_validation_sum += loss.item()
                            loss_validation = loss_validation_sum / len(dataloader_validation)
                            loss_validation_array[experiment_name_index, architecture_index, encoder_name_index, encoder_weights_index, epoch_index] = loss_validation
                            if loss_validation < loss_validation_best:
                                loss_validation_best = loss_validation
                                torch.save(model.state_dict(), model_file_path)
                    model = architecture(encoder_name, encoder_weights=encoder_weights, activation="sigmoid", in_channels=1).to(device)
                    model.load_state_dict(torch.load(model_file_path))
                    slices_test_num = 0
                    for index_test_volume in range_test_volume:
                        dataset_test = MedicalSegmentation2(index_test_volume, use_transforms=False)
                        dataloader_test = DataLoader(dataset_test)
                        slices_test_num += len(dataloader_test)
                        model.eval()
                        with torch.no_grad():
                            for images, mask_lungs, mask_lesions in dataloader_test:
                                if experiment_name == experiment_names[0]:
                                    masks = mask_lungs
                                elif experiment_name == experiment_names[1]:
                                    masks = mask_lesions
                                elif experiment_name == experiment_names[2]:
                                    images *= mask_lungs # noqa: PLW2901
                                    masks = mask_lesions
                                images = images.to(device) # noqa: PLW2901
                                masks = masks.to(device)
                                predictions = model(images)
                                if architecture_name == architecture_names[0] and encoder_name == encoder_names[0] and (encoder_weights == encoders_weights[0]):
                                    hist_images_array[experiment_name_index] += np.histogram(images.cpu(), hist_bins, hist_range)[0]
                                    hist_masks_array[experiment_name_index] += np.histogram(images.cpu() * masks.cpu(), hist_bins, hist_range)[0]
                                metrics_values = calculate_metrics(masks, predictions)
                                metrics_array[experiment_name_index, architecture_index, encoder_name_index, encoder_weights_index] += metrics_values
                            if index_test_volume == 0:
                                index_test = 12
                                images = dataset_test[index_test][0]
                                mask_lungs = dataset_test[index_test][1]
                                mask_lesions = dataset_test[index_test][2]
                                if experiment_name == experiment_names[0]:
                                    masks = mask_lungs
                                elif experiment_name == experiment_names[1]:
                                    masks = mask_lesions
                                elif experiment_name == experiment_names[2]:
                                    images *= mask_lungs
                                    masks = mask_lesions
                                predictions = model(images.unsqueeze(0).to(device))
                                save_figure_image_masked(architecture_name, encoder_name, experiment_name, images[0], masks[0], predictions[0, 0])
                            if index_test_volume == 0 and encoder_name == "resnet18" and (encoder_weights is None):
                                volume_mask_array = np.zeros((512, 512, len(dataset_test)))
                                volume_prediction_array = np.zeros((512, 512, len(dataset_test)))
                                for slice_volume_index, (images, mask_lungs, mask_lesions) in enumerate(dataset_test): # type: ignore[arg-type, misc]
                                    if experiment_name == experiment_names[0]:
                                        masks = mask_lungs
                                    elif experiment_name == experiment_names[1]:
                                        masks = mask_lesions
                                    elif experiment_name == experiment_names[2]:
                                        images *= mask_lungs # noqa: PLW2901
                                        masks = mask_lesions
                                    volume_mask_array[:, :, slice_volume_index] = masks[0].float()
                                    predictions = model(images.unsqueeze(0).to(device))
                                    volume_prediction_array[:, :, slice_volume_index] = predictions[0].float().cpu()
                                volume_mask_array = volume_mask_array[:, :, ::-1]
                                if architecture_name == "Unet":
                                    save_figure_3d("mask", "", experiment_name, step_size, volume_mask_array)
                                volume_prediction_array = volume_prediction_array[:, :, ::-1]
                                save_figure_3d(architecture_name, encoder_weights, experiment_name, step_size, volume_prediction_array)
                    model_file_name = f"{experiment_name}-{architecture_name}-{encoder_name}-{encoder_weights}"
                    if model_file_name in ["lesion-segmentation-a-FPN-mobilenet_v2-imagenet", "lung-segmentation-FPN-mobilenet_v2-imagenet"]:
                        save_tfjs_from_torch(dataset_train[0][0].unsqueeze(0), model, model_file_name)
                        if environ["DEBUG"] != "1":
                            dist_path = Path("dist") / model_file_name
                            rmtree(dist_path)
                            move(f"bin/{model_file_name}", dist_path)
                    if environ["DEBUG"] == "1":
                        model_file_path.unlink()
    for hist_images, hist_masks, experiment_name in zip(hist_images_array, hist_masks_array, experiment_names, strict=True):
        save_figure_histogram(experiment_name, hist_images, hist_masks, hist_range)
    for experiment_name, loss_train_, loss_validation_ in zip(experiment_names, loss_train_array, loss_validation_array, strict=True):
        save_figure_loss(architecture_names, experiment_name, loss_train_, "Train", [0, 1])
        save_figure_loss(architecture_names, experiment_name, loss_validation_, "Validation", [0, 1])
    save_figure_loss(architecture_names, experiment_name, loss_train_array[2] - loss_train_array[1], "Train diff", [-0.4, 0.4])
    save_figure_loss(architecture_names, experiment_name, loss_validation_array[2] - loss_validation_array[1], "Validation diff", [-0.4, 0.4])
    metrics_array = 100 * np.nan_to_num(metrics_array) / slices_test_num
    parameters_num_array = parameters_num_array / 10 ** 6
    for experiment_name, metrics_ in zip(experiment_names, metrics_array, strict=True):
        save_figure_architecture_box(architecture_names, metrics_[..., -1], experiment_name)
        save_figure_scatter(architecture_names, metrics_[..., -1], experiment_name, parameters_num_array, [70, 100])
    save_figure_scatter(architecture_names, metrics_array[2, ..., -1] - metrics_array[1, ..., -1], "diff", parameters_num_array, [-15, 15])
    save_figure_initialization_box(metrics_array[..., -1], encoders_weights)
    encoder_weights_mean = metrics_array[..., -1].reshape(-1, len(encoders_weights)).mean(0).round(1)
    encoder_mean = metrics_array.transpose([2, 0, 1, 3, 4]).reshape(metrics_array.shape[2], -1).mean(1)
    metrics_array = metrics_array.transpose([1, 2, 3, 0, 4])
    metrics_array_global_mean = metrics_array.reshape(-1, np.prod(metrics_array.shape[2:])).mean(0)
    metrics_array = np.concatenate((metrics_array, metrics_array.mean(1, keepdims=True)), 1)
    metrics_array = metrics_array.reshape(-1, np.prod(metrics_array.shape[2:]))
    parameters_num_array_global_mean = parameters_num_array.mean()
    parameters_num_array = np.concatenate((parameters_num_array, parameters_num_array.mean(1, keepdims=True)), 1)
    flops_array /= 10 ** 9
    flops_array = flops_array.mean(0).mean(-1)
    flops_array_global_mean = flops_array.mean()
    flops_array = np.concatenate((flops_array, flops_array.mean(1, keepdims=True)), 1)
    index = pd.MultiIndex.from_product([architecture_names, [encoder_name.replace("_", "") for encoder_name in encoder_names] + ["Mean"]])
    multicolumn = pd.MultiIndex.from_product([[str(encoder_weights) for encoder_weights in encoders_weights], experiments, metric_names])
    metrics_df = pd.DataFrame(metrics_array, index=index, columns=multicolumn)
    metrics_df["Performance", "related", "Pars (M)"] = parameters_num_array.flatten()
    metrics_df["Performance", "related", "FLOPS (B)"] = flops_array.flatten()
    metrics_df.loc[("Global", "Mean"), :] = np.append(metrics_array_global_mean, (parameters_num_array_global_mean, flops_array_global_mean))
    metrics_df.index.names = ["Architecture", "Encoder"]
    metrics_df = metrics_df.round(1)
    maxs_per_column = metrics_df.max(0)
    maxs_per_column_index = metrics_df.idxmax(0)
    styler = metrics_df.style
    styler.format(precision=1)
    styler.highlight_max(props="bfseries: ;")
    styler.highlight_min(props="bfseries: ;")
    styler.to_latex("bin/metrics.tex", hrules=True, multicol_align="c")
    keys = ["epochs-num", "batch-size", "test-slices-num", "encoder-best", "encoder-worst"] + [f"{experiment_name}-{encoder_weights}-mean" for experiment_name, encoder_weights in itertools.product(experiment_names, encoders_weights)] + [f"{experiment_name}-{encoder_weights}-max" for experiment_name, encoder_weights in itertools.product(experiment_names, encoders_weights)] + [f"{encoder_weights}-{stat}" for encoder_weights, stat in itertools.product(encoders_weights, ["mean"])] + [f"{experiment_name}-architecture-{encoder_weights}-index-max" for experiment_name, encoder_weights in itertools.product(experiment_names, encoders_weights)] + [f"{experiment_name}-encoder-{encoder_weights}-index-max" for experiment_name, encoder_weights in itertools.product(experiment_names, encoders_weights)] + [f"{experiment_name}-{architecture_name}-{encoder_weights}-mean" for experiment_name, architecture_name, encoder_weights in itertools.product(experiment_names, architecture_names, encoders_weights)]
    values = [str(int(epochs_num)), str(int(batch_size)), str(int(slices_test_num)), encoder_names[encoder_mean.argmax()].replace("_", ""), encoder_names[encoder_mean.argmin()].replace("_", "")] + [metrics_df.loc["Global", "Mean"][str(encoder_weights), experiment]["Dice"] for experiment, encoder_weights in itertools.product(experiments, encoders_weights)] + [maxs_per_column[2], maxs_per_column[11], maxs_per_column[5], maxs_per_column[14], maxs_per_column[8], maxs_per_column[17], encoder_weights_mean[0], encoder_weights_mean[1], maxs_per_column_index[2][0], maxs_per_column_index[2][1], maxs_per_column_index[11][0], maxs_per_column_index[11][1], maxs_per_column_index[5][0], maxs_per_column_index[5][1], maxs_per_column_index[14][0], maxs_per_column_index[14][1], maxs_per_column_index[8][0], maxs_per_column_index[8][1], maxs_per_column_index[17][0], maxs_per_column_index[17][1]] + [metrics_df.loc[architecture_name, "Mean"][str(encoder_weights), experiment]["Dice"] for experiment, architecture_name, encoder_weights in itertools.product(experiments, architecture_names, encoders_weights)]
    keys_values_df = pd.DataFrame({"key": keys, "value": values})
    keys_values_df.to_csv("bin/keys-values.csv")


def preprocess_image(image: np.ndarray, mask_lesion: np.ndarray, mask_lung: np.ndarray, use_transforms: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # type: ignore[type-arg] # noqa: FBT001
    image = tf.to_pil_image(image.astype("float32"))
    mask_lesion = tf.to_pil_image(mask_lesion.astype("uint8"))
    mask_lung = tf.to_pil_image(mask_lung.astype("uint8"))
    image = tf.resize(image, [512, 512])
    mask_lesion = tf.resize(mask_lesion, [512, 512])
    mask_lung = tf.resize(mask_lung, [512, 512])
    if use_transforms:
        if rng.random() > 0.5: # noqa: PLR2004
            image = tf.hflip(image)
            mask_lesion = tf.hflip(mask_lesion)
            mask_lung = tf.hflip(mask_lung)
        if rng.random() > 0.5: # noqa: PLR2004
            image = tf.vflip(image)
            mask_lesion = tf.vflip(mask_lesion)
            mask_lung = tf.vflip(mask_lung)
        scale = rng.random() + 0.5
        rotation = 360 * rng.random() - 180
        image = tf.affine(image, rotation, [0, 0], scale, 0)
        mask_lesion = tf.affine(mask_lesion, rotation, [0, 0], scale, 0)
        mask_lung = tf.affine(mask_lung, rotation, [0, 0], scale, 0)
    image = tf.to_tensor(image)/4095
    mask_lesion = tf.to_tensor(mask_lesion).bool()
    mask_lung = tf.to_tensor(mask_lung).bool()
    return (image, mask_lung, mask_lesion)


def save_figure_3d(architecture: str, encoder_weights: str | None, experiment_name: str, step_size: int, volume: np.ndarray) -> None: # type: ignore[type-arg]
    volume = volume > 0.5 # noqa: PLR2004
    volume[0, 0, 0:10] = 0
    volume[0, 0, 10:20] = 1
    verts, faces, *_ = marching_cubes(volume, 0.5, step_size=step_size)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.5, rasterized=True)
    ax.set_xlim([1, volume.shape[0]])
    ax.set_ylim([1, volume.shape[1]])
    ax.set_zlim([1, volume.shape[2]])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False) # noqa: FBT003
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False) # noqa: FBT003
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False) # noqa: FBT003
    plt.savefig(f"bin/{experiment_name}-{architecture}-{encoder_weights}-volume")
    plt.close()


def save_figure_architecture_box(architecture_names: list[str], dice: np.ndarray, experiment_name: str) -> None: # type: ignore[type-arg]
    dice_ = dice.reshape(dice.shape[0], -1).T
    plt.subplots()
    plt.boxplot(dice_)
    plt.grid(visible=True)
    plt.xticks(ticks=np.arange(len(architecture_names)) + 1, labels=architecture_names, fontsize=15)
    plt.ylim([70, 100])
    plt.savefig(f"bin/{experiment_name}-boxplot-dice")
    plt.close()


def save_figure_histogram(experiment_name: str, hist_images: np.ndarray, hist_masks: np.ndarray, hist_range: tuple[float, float]) -> None: # type: ignore[type-arg]
    t_linspace_array = np.linspace(hist_range[0], hist_range[1], hist_masks.shape[-1])
    hist_images = hist_images.reshape(-1, hist_images.shape[-1]).sum(0)
    hist_masks = hist_masks.reshape(-1, hist_masks.shape[-1]).sum(0)
    hist_maxes = max([hist_images.max(), hist_masks.max()])
    hist_images /= hist_maxes
    hist_masks /= hist_maxes
    _, ax = plt.subplots()
    plt.bar(t_linspace_array, hist_images, width=t_linspace_array[1] - t_linspace_array[0], align="center", label="Images")
    plt.bar(t_linspace_array, hist_masks, width=t_linspace_array[1] - t_linspace_array[0], align="center", label="Masks")
    plt.xlabel("Normalized values", fontsize=15)
    plt.xlim(hist_range)
    plt.ylim([10 ** (-7), 1])
    plt.grid(visible=True, which="both")
    ax.set_yscale("log")
    ax.legend()
    plt.savefig(f"bin/{experiment_name}-hist")
    plt.close()


def save_figure_image(experiment_name: str, image: torch.Tensor) -> None:
    image = image.cpu().numpy()
    _, ax = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(image, cmap="gray", vmin=-0.5, vmax=0.5)
    plt.savefig(f"bin/{experiment_name}-image")
    plt.close()


def save_figure_image_masked(architecture: str, encoder_name: str, experiment_name: str, image: torch.Tensor, mask: torch.Tensor, prediction: torch.Tensor) -> None: # noqa: PLR0913
    image = image.cpu().numpy()
    mask = mask.cpu().numpy()
    prediction = prediction.cpu().detach().numpy()
    prediction = prediction > 0.5 # noqa: PLR2004
    correct = mask * prediction
    false = np.logical_xor(mask, prediction) > 0.5 # noqa: PLR2004
    _, ax = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(image, cmap="gray", vmin=-0.5, vmax=0.5)
    plt.imshow(correct, cmap="Greens", alpha=0.3)
    plt.imshow(false, cmap="Reds", alpha=0.3)
    plt.savefig(f"bin/{experiment_name}-{architecture}-{encoder_name.replace('_', '')}-masked-image")
    plt.close()


def save_figure_initialization_box(dice: np.ndarray, encoders_weights: list[str | None]) -> None: # type: ignore[type-arg]
    dice_ = dice.reshape(-1, dice.shape[-1])
    plt.subplots()
    plt.boxplot(dice_)
    plt.grid(visible=True)
    plt.xticks(ticks=np.arange(len(encoders_weights)) + 1, labels=[str(encoder_weights) for encoder_weights in encoders_weights], fontsize=15)
    plt.ylim([70, 100])
    plt.savefig("bin/initialization-boxplot-dice")
    plt.close()


def save_figure_loss(architecture_names: list[str], experiment_name: str, loss: np.ndarray, train_or_validation: str, ylim: list[float]) -> None: # type: ignore[type-arg]
    loss = np.nan_to_num(loss)
    p1 = []
    p2 = []
    t_range_array = np.arange(1, loss.shape[-1] + 1)
    loss = loss.reshape(loss.shape[0], -1, loss.shape[-1])
    loss_mean_list = loss.mean(1)
    loss_std_list = loss.std(1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(loss_mean_list)]
    _, ax = plt.subplots()
    for _index, (loss_mean, loss_std, color) in enumerate(zip(loss_mean_list, loss_std_list, colors, strict=True)):
        ax.fill_between(t_range_array, loss_mean + loss_std, loss_mean - loss_std, facecolor=color, alpha=0.3)
        p1.append(ax.plot(t_range_array, loss_mean, color=color))
        p2.append(ax.fill(np.nan, np.nan, color, alpha=0.3))
    ax.legend([(p2[0][0], p1[0][0]), (p2[1][0], p1[1][0]), (p2[2][0], p1[2][0]), (p2[3][0], p1[3][0])], architecture_names, loc="upper right")
    plt.grid(visible=True)
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.ylim(ylim)
    if train_or_validation not in ["Train", "Validation"]:
        plt.xlabel("Epochs", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize="large")
    ax.tick_params(axis="both", which="minor", labelsize="large")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"bin/{experiment_name}-{train_or_validation.lower().replace(' ', '-')}-loss")
    plt.close()


def save_figure_scatter(architecture_names: list[str], dice: np.ndarray, experiment_name: str, parameters_num_array: np.ndarray, ylim: list[int]) -> None: # type: ignore[type-arg]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(parameters_num_array)]
    dice_ = dice.mean(-1)
    _, ax = plt.subplots()
    for parameters_num, dice_element, architecture_name, color in zip(parameters_num_array, dice_, architecture_names, colors, strict=True):
        plt.scatter(parameters_num, dice_element, c=color, label=architecture_name, s=3)
    xmin = 0
    xmax = 80
    ymin = ylim[0]
    ymax = ylim[1]
    nbins = 100
    x_mgrid, y_mgrid = np.mgrid[xmin:xmax:nbins * 1j, ymin:ymax:nbins * 1j] # type: ignore[misc]
    positions = np.vstack([x_mgrid.ravel(), y_mgrid.ravel()])
    values = np.vstack([parameters_num_array.flatten(), dice_.flatten()])
    kernel = gaussian_kde(values)
    z_grid = np.reshape(kernel(positions).T, x_mgrid.shape)
    ax.imshow(np.rot90(z_grid), cmap="Greens", extent=[xmin, xmax, ymin, ymax], alpha=0.5)
    plt.grid(visible=True)
    plt.xlabel("Number of parameters ($10^6$)", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize="large")
    ax.tick_params(axis="both", which="minor", labelsize="large")
    plt.xlim([xmin, xmax])
    plt.ylim(ylim)
    ax.legend(loc="lower right")
    ax.set_aspect(aspect="auto")
    plt.savefig(f"bin/{experiment_name}-scatter-dice-vs-num-parameters")
    plt.close()


def save_figure_weights(architecture_name: str, encoder_weights: str, experiment_name: str, model: nn.Module) -> None:
    rows = 8
    columns = 8
    plt.figure(figsize=(4, 4.6))
    gs = gridspec.GridSpec(rows, columns, wspace=0.0, hspace=0.0, left=0)
    for rows_index in range(rows):
        for columns_index in range(columns):
            weight = next(model.children()).conv1.weight[rows_index * rows + columns_index] # type: ignore[index, union-attr]
            ax = plt.subplot(gs[rows_index, columns_index])
            plt.imshow(weight[0].detach().cpu().numpy(), cmap="gray", vmin=-0.4, vmax=0.4)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.savefig(f"bin/{experiment_name}-{architecture_name}-{encoder_weights}-weights")
    plt.close()


def save_tfjs_from_torch(example_input: torch.Tensor, model: nn.Module, model_file_name: str) -> None:
    model_file_path = Path("bin") / model_file_name
    if model_file_path.exists():
        rmtree(model_file_path)
    model_file_path.mkdir()
    torch.onnx.export(model.cpu(), example_input, model_file_path / "model.onnx", export_params=True, opset_version=11)
    model_onnx = onnx.load(model_file_path / "model.onnx")
    model_tf = prepare(model_onnx)
    model_tf.export_graph(model_file_path / "model")
    tf_saved_model_conversion_v2.convert_tf_saved_model((model_file_path / "model").as_posix(), model_file_path, skip_op_check=True)
    rmtree(model_file_path / "model")
    (model_file_path / "model.onnx").unlink()


if __name__ == "__main__":
    main()
