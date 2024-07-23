# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'


if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import torch
import time
import pickle
import gzip
import glob

try:
    from settings import *
except:
    from .settings import *

def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), protocol=4)


def fuze_models(
        best_weights,
        output_path
):
    result = []
    for path in best_weights:
        print('Read: {}'.format(path))
        res = torch.load(path, map_location='cpu')
        result.append(res)
    save_in_file_fast(result, output_path)


def find_best_weights(folder, num_folds=5):
    all_weights = []
    for fold in range(num_folds):
        fold_weights = glob.glob(folder + '*_fold_{}_*.pth'.format(fold))
        min_mae = 100000000.0
        best_weight = None
        for f in fold_weights:
            if '_mae_' not in f:
                continue
            r = f.split('_mae_')[1]
            r = r.split('_')[0]
            r = float(r)
            if r < min_mae:
                best_weight = f
                min_mae = r
        all_weights.append(best_weight)
    return all_weights


if __name__ == '__main__':
    start_time = time.time()

    MODELS_PATH_TORCH = MODELS_PATH + 'training_r/'
    best_weights = find_best_weights(MODELS_PATH_TORCH)
    print('Best weights for folds:')
    for i, b in enumerate(best_weights):
        print("Fold {}: {}".format(i, b))
    output_path = MODELS_PATH_TORCH + 'fuzed_model.pkl'

    fuze_models(
        best_weights,
        output_path
    )
    print("Fused model saved in: {}".format(output_path))
    print('Time: {:.0f} sec'.format(time.time() - start_time))
