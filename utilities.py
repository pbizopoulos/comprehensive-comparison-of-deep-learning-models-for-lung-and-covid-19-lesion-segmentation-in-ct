import numpy as np

from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from skimage.measure import marching_cubes

plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'


def save_weights(model, experiment_name, architecture_name, encoder_weights):
    rows = 8
    columns = 8
    fig = plt.figure(figsize=(4, 4.6))
    gs = gridspec.GridSpec(rows, columns, wspace=0.0, hspace=0.0, left=0)
    for index_rows in range(rows):
        for index_columns in range(columns):
            weight = list(model.children())[0].conv1.weight[index_rows*rows+index_columns]
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
    fig, ax = plt.subplots()
    plt.bar(t, hist_images, width=t[1]-t[0], align='center', label='Images')
    plt.bar(t, hist_masks, width=t[1]-t[0], align='center', label='Masks')
    plt.xlabel('Normalized values', fontsize=15)
    plt.xlim(hist_range)
    plt.ylim([10**(-7), 1])
    plt.grid(True, which='both')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(f'tmp/{experiment_name}-hist')
    plt.close()

def save_loss(loss, train_or_validation, experiment_name, architecture_name_list, ylim):
    loss = np.nan_to_num(loss)
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    p1 = [None]*len(architecture_name_list)
    p2 = [None]*len(architecture_name_list)
    t = np.arange(1, loss.shape[-1] + 1)
    loss = loss.reshape(loss.shape[0], -1, loss.shape[-1])
    mus = loss.mean(1)
    sigmas = loss.std(1)
    fig, ax = plt.subplots()
    for index, (mu, sigma, color) in enumerate(zip(mus, sigmas, color_list)):
        ax.fill_between(t, mu+sigma, mu-sigma, facecolor=color, alpha=0.3)
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
    fig, ax = plt.subplots()
    plt.boxplot(dice_)
    plt.grid(True)
    plt.xticks(ticks=np.arange(len(architecture_name_list))+1, labels=architecture_name_list, fontsize=15)
    plt.ylim([70, 100])
    plt.savefig(f'tmp/{experiment_name}-boxplot-dice')
    plt.close()

def save_initialization_box(dice, encoder_weights_list):
    dice_ = dice.reshape(-1, dice.shape[-1])
    fig, ax = plt.subplots()
    plt.boxplot(dice_)
    plt.grid(True)
    plt.xticks(ticks=np.arange(len(encoder_weights_list))+1, labels=[str(e) for e in encoder_weights_list], fontsize=15)
    plt.ylim([70, 100])
    plt.savefig(f'tmp/initialization-boxplot-dice')
    plt.close()

def save_scatter(num_parameters_array, dice, architecture_name_list, experiment_name, ylim):
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    dice_ = dice.mean(-1)
    fig, ax = plt.subplots()
    for num_parameters, d, architecture_name, color in zip(num_parameters_array, dice_, architecture_name_list, color_list):
        plt.scatter(num_parameters, d, c=color, label=architecture_name, s=3)
    x = num_parameters_array.flatten()
    y = dice_.flatten()
    xmin = 0
    xmax = 80
    ymin = ylim[0]
    ymax = ylim[1]
    nbins = 100
    X, Y = np.mgrid[xmin:xmax:nbins*1j, ymin:ymax:nbins*1j]
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
    fig, ax = plt.subplots()
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
    fig, ax = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(image, cmap='gray', vmin=-1.5, vmax=1.5)
    plt.imshow(correct, cmap='Greens', alpha=0.3)
    plt.imshow(false, cmap='Reds', alpha=0.3)
    plt.savefig(f'tmp/{experiment_name}-{architecture}-{encoder.replace("_", "")}-masked-image')
    plt.close()

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
