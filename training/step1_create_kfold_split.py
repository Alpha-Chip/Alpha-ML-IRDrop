# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

try:
    from settings import *
except:
    from .settings import *


def create_random_kfold_split(num_folds, seed):
    np.random.seed(seed)
    kfold_file = OUTPUT_PATH + 'kfold_split_{}_seed_{}.csv'.format(num_folds, seed)
    if os.path.isfile(kfold_file):
        print('Already exists! {}'.format(kfold_file))
        exit()

    test_folders = glob.glob(INPUT_PATH + 'real-circuit-data/*')
    data = [os.path.basename(f) for f in test_folders]
    print(data)
    kf = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

    folds = np.zeros(len(data), dtype=np.int32)
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        folds[test_index] = i

    s = pd.DataFrame(data, columns=['id'])
    s['fold'] = folds
    s.to_csv(kfold_file, index=False)
    print(s)


if __name__ == '__main__':
    create_random_kfold_split(5, 42)

