import argparse
import itertools
import numpy as np
import pandas as pd
import torch

from torch import optim
from torch.utils.data import DataLoader

from segmentation_models_pytorch import Unet, Linknet, FPN, PSPNet

from datasets import MedicalSegmentation1, MedicalSegmentation2, CTSegBenchmark
from losses import dice_loss, metrics
from utilities import save_weights, save_hist, save_loss, save_scatter, save_architecture_box, save_initialization_box, save_3d, save_image, save_masked_image, get_num_parameters


if __name__ == '__main__':
    # DO NOT EDIT BLOCK - Required by the Makefile
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    parser.add_argument('tmp_dir')
    parser.add_argument('--full', default=False, action='store_true')
    args = parser.parse_args()
    # END OF DO NOT EDIT BLOCK

    # Set random seeds for reproducibility.
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    batch_size = 2
    architecture_list = [Unet, Linknet, FPN, PSPNet]
    architecture_name_list = [architecture.__name__ for architecture in architecture_list]
    experiment_list = ['Lung segmentation', 'Lesion segmentation A', 'Lesion segmentation B']
    experiment_name_list = [experiment.lower().replace(' ', '-') for experiment in experiment_list]
    encoder_weights_list = [None, 'imagenet']
    train_dataset = MedicalSegmentation1(args.tmp_dir, index_train_range, use_transforms=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = MedicalSegmentation1(args.tmp_dir, index_validation_range, use_transforms=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
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
                    model_path = f'{args.tmp_dir}/{experiment_name}-{architecture_name}-{encoder}-{encoder_weights}.pt'
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
                                save_image(images[0, 0], experiment_name, args.results_dir)
                                save_masked_image(images[0, 0], masks[0, 0], masks[0, 0], experiment_name, 'mask', 'train', args.results_dir)
                                save_masked_image(images[0, 0], masks[0, 0], predictions[0, 0], experiment_name, 'prediction', 'train', args.results_dir)
                            if (architecture_name == architecture_name_list[0]) and (encoder == 'resnet18'):
                                if (index_epoch == 0):
                                    save_weights(model, experiment_name, architecture_name, f'{encoder_weights}-before', args.results_dir)
                                elif (index_epoch == num_epochs - 1):
                                    save_weights(model, experiment_name, architecture_name, f'{encoder_weights}-after', args.results_dir)
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
                        test_dataset = MedicalSegmentation2(args.tmp_dir, index_test_volume, use_transforms=False)
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
                                save_masked_image(images[0], masks[0], predictions[0, 0], experiment_name, architecture_name, encoder, args.results_dir)
                            if (index_test_volume == 0) and (encoder == 'resnet18') and (encoder_weights == None):
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
                                    save_3d(volume_mask, args.full, experiment_name, 'mask', '', args.results_dir)
                                volume_prediction = volume_prediction[:, :, ::-1]
                                save_3d(volume_prediction, args.full, experiment_name, architecture_name, encoder_weights, args.results_dir)

    for hist_images, hist_masks, experiment_name in zip(hist_images_array, hist_masks_array, experiment_name_list):
        save_hist(hist_images, hist_masks, hist_range, experiment_name, args.results_dir)

    for experiment_name, train_loss, validation_loss in zip(experiment_name_list, train_loss_array, validation_loss_array):
        save_loss(train_loss, 'Train', experiment_name, architecture_name_list, [0, 1], args.results_dir)
        save_loss(validation_loss, 'Validation', experiment_name, architecture_name_list, [0, 1], args.results_dir)
    save_loss(train_loss_array[2] - train_loss_array[1], 'Train diff', experiment_name, architecture_name_list, [-0.4, 0.4], args.results_dir)
    save_loss(validation_loss_array[2] - validation_loss_array[1], 'Validation diff', experiment_name, architecture_name_list, [-0.4, 0.4], args.results_dir)

    metrics_array = 100*metrics_array/num_slices_test
    num_parameters_array = num_parameters_array/10**6
    for experiment, experiment_name, metrics_ in zip(experiment_list, experiment_name_list, metrics_array):
        save_architecture_box(metrics_[..., -1], architecture_name_list, experiment_name, args.results_dir)
        save_scatter(num_parameters_array, metrics_[..., -1], architecture_name_list, experiment_name, [70, 100], args.results_dir)
    save_scatter(num_parameters_array, metrics_array[2, ..., -1] - metrics_array[1, ..., -1], architecture_name_list, 'diff', [-15, 15], args.results_dir)
    save_initialization_box(metrics_array[..., -1], encoder_weights_list, args.results_dir)

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

    train_time_array /= (num_epochs * 10**6)
    train_time_array = train_time_array.mean(0).mean(-1)
    train_time_array_global_mean = train_time_array.mean()
    train_time_array_global_std = train_time_array.std()
    train_time_array = np.concatenate((train_time_array, train_time_array.mean(1, keepdims=True), train_time_array.std(1, keepdims=True)), 1)

    validation_time_array /= (num_epochs * 10**6)
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
    formatters = [lambda x,max_per_column=max_per_column: fr'\bf{{{x:.2f}}}' if (x == max_per_column) else f'{x:.2f}' for max_per_column in max_per_column_list]
    formatters[-3:] = [lambda x,min_per_column=min_per_column: fr'\bf{{{x:.2f}}}' if (x == min_per_column) else f'{x:.2f}' for min_per_column in min_per_column_list[-3:]]
    df.to_latex(f'{args.results_dir}/metrics.tex', formatters=formatters, bold_rows=True, column_format='c|l|rrr|rrr|rrr|rrr|rrr|rrr|rrr', multirow=True, multicolumn=True, multicolumn_format='c', escape=False)

    df_keys_values = pd.DataFrame({'key': [
        'num_epochs',
        'batch_size',
        'num_slices_test',
        'encoder-best',
        'encoder-worst'] + \
        [f'{experiment_name}-{encoder_weights}-mean' for experiment_name, encoder_weights in itertools.product(experiment_name_list, encoder_weights_list)] + \
        [f'{experiment_name}-{encoder_weights}-std' for experiment_name, encoder_weights in itertools.product(experiment_name_list, encoder_weights_list)] + \
        [f'{experiment_name}-{encoder_weights}-max' for experiment_name, encoder_weights in itertools.product(experiment_name_list, encoder_weights_list)] + \
        [f'{encoder_weights}-{stat}' for encoder_weights, stat in itertools.product(encoder_weights_list, ['mean', 'std'])] + \
        [f'{experiment_name}-architecture-{encoder_weights}-index-max' for experiment_name, encoder_weights in itertools.product(experiment_name_list, encoder_weights_list)] + \
        [f'{experiment_name}-encoder-{encoder_weights}-index-max' for experiment_name, encoder_weights in itertools.product(experiment_name_list, encoder_weights_list)] + \
        [f'{experiment_name}-{architecture_name}-{encoder_weights}-mean' for experiment_name, architecture_name, encoder_weights in itertools.product(experiment_name_list, architecture_name_list, encoder_weights_list)] + \
        [f'{experiment_name}-{architecture_name}-{encoder_weights}-std' for experiment_name, architecture_name, encoder_weights in itertools.product(experiment_name_list, architecture_name_list, encoder_weights_list)],
        'value': [
            str(int(num_epochs)),
            str(int(batch_size)),
            str(int(num_slices_test)),
            encoder_list[encoder_mean.argmax()].replace('_', ''),
            encoder_list[encoder_mean.argmin()].replace('_', '')] + \
            [df.loc['Global', 'Mean'][str(encoder_weights), experiment]['Dice'] for experiment, encoder_weights in itertools.product(experiment_list, encoder_weights_list)] + \
            [df.loc['Global', 'Std'][str(encoder_weights), experiment]['Dice'] for experiment, encoder_weights in itertools.product(experiment_list, encoder_weights_list)] + \
            [max_per_column_list[2],
            max_per_column_list[11],
            max_per_column_list[5],
            max_per_column_list[14],
            max_per_column_list[8],
            max_per_column_list[17],
            encoder_weights_mean[0],
            encoder_weights_mean[1],
            encoder_weights_std[0],
            encoder_weights_std[1],
            index_max_per_column_list[2][0],
            index_max_per_column_list[2][1],
            index_max_per_column_list[11][0],
            index_max_per_column_list[11][1],
            index_max_per_column_list[5][0],
            index_max_per_column_list[5][1],
            index_max_per_column_list[14][0],
            index_max_per_column_list[14][1],
            index_max_per_column_list[8][0],
            index_max_per_column_list[8][1],
            index_max_per_column_list[17][0],
            index_max_per_column_list[17][1]] + \
            [df.loc[architecture_name, 'Mean'][str(encoder_weights), experiment]['Dice'] for experiment, architecture_name, encoder_weights in itertools.product(experiment_list, architecture_name_list, encoder_weights_list)] + \
            [df.loc[architecture_name, 'Std'][str(encoder_weights), experiment]['Dice'] for experiment, architecture_name, encoder_weights in itertools.product(experiment_list, architecture_name_list, encoder_weights_list)]
            })
    df_keys_values.to_csv(f'{args.results_dir}/keys-values.csv')
