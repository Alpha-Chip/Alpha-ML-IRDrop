# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


def draw_irdrop(opt):
    # print(list(colormaps))
    real = opt['real']
    pred = opt['pred']
    output_file = opt['output']

    ir_drop_map_real = pd.read_csv(real, header=None).values * 100000
    ir_drop_map_pred = pd.read_csv(pred, header=None).values * 100000

    print('Shape real: {}'.format(ir_drop_map_real.shape))
    print('Shape pred: {}'.format(ir_drop_map_pred.shape))

    mae_value = np.abs(ir_drop_map_pred - ir_drop_map_real).sum() / len(ir_drop_map_real.flatten())

    fig = plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title('Real')
    plt.imshow(ir_drop_map_real, interpolation='none', cmap=colormaps['turbo'])
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title('Pred [MAE: {:.4f}]'.format(mae_value))
    plt.imshow(ir_drop_map_pred, interpolation='none', cmap=colormaps['turbo'])
    plt.colorbar()

    plt.savefig(output_file)
    plt.close(fig)
    print('Image was saved in: {}'.format(output_file))


if __name__ == '__main__':
    m = argparse.ArgumentParser()
    m.add_argument("--real", "-r", type=str, help="Real IRDrop map in CSV-format")
    m.add_argument("--pred", "-p", type=str, help="Predicted IRDrop map in CSV-format", required=True)
    m.add_argument("--output", "-o", type=str, help="Path to output image file", required=True)
    options = m.parse_args().__dict__
    print(options)
    draw_irdrop(options)