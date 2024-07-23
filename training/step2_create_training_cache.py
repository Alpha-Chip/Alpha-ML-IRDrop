# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'

import os
import glob
import pandas as pd
import pickle
from utils.converter import expand_matrix

try:
    from settings import *
except:
    from .settings import *


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), protocol=4)


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def create_cache_file(out_dir):
    res = dict()
    res['fake'] = dict()
    res['real'] = dict()

    if os.path.isdir(INPUT_PATH + "designs_Ilya/fakes/"):
        files = glob.glob(INPUT_PATH + "designs_Ilya/fakes/*/current_map.csv")
        print("Fake files found:", len(files))
        for f in files:
            id = os.path.basename(os.path.dirname(f))
            print(f, "ID", id)
            current = pd.read_csv(f, header=None).values
            eff_dist = pd.read_csv(os.path.dirname(f) + '/eff_dist_map.csv', header=None).values
            pdn_density = pd.read_csv(os.path.dirname(f) + '/pdn_density.csv', header=None).values
            ir_drop_map = pd.read_csv(os.path.dirname(f) + '/ir_drop_map.csv', header=None).values
            print(current.shape, eff_dist.shape, pdn_density.shape, ir_drop_map.shape)
            if not (current.shape == eff_dist.shape == pdn_density.shape == ir_drop_map.shape):
                print('Not equal!')
                ir_drop_map = expand_matrix(ir_drop_map, current.shape[0], current.shape[1])

            out_file_name = out_dir + 'fake_type_0_{}'.format(id) + '.pkl'
            save_in_file_fast([
                100000000 * current,
                100 * eff_dist,
                100 * pdn_density,
                100000 * ir_drop_map
            ], out_file_name)
    else:
        print('Warning! Folder not found: {}. It will not be used for training!'.format(INPUT_PATH + "designs_Ilya/fakes/"))

    if os.path.isdir(INPUT_PATH + "began_processed/"):
        files = glob.glob(INPUT_PATH + "began_processed/*/current_map.csv")
        print("BeGAN files found:", len(files))
        for f in files:
            id = os.path.basename(os.path.dirname(f))
            print(f, "ID", id)
            current = pd.read_csv(f, header=None).values
            eff_dist = pd.read_csv(os.path.dirname(f) + '/eff_dist_map.csv', header=None).values
            pdn_density = pd.read_csv(os.path.dirname(f) + '/pdn_density.csv', header=None).values
            ir_drop_map = pd.read_csv(os.path.dirname(f) + '/ir_drop_map.csv', header=None).values
            print(current.shape, eff_dist.shape, pdn_density.shape, ir_drop_map.shape)
            if not (current.shape == eff_dist.shape == pdn_density.shape == ir_drop_map.shape):
                print('Not equal!')
                exit()

            out_file_name = out_dir + 'fake_type_1_{}'.format(id) + '.pkl'
            save_in_file_fast([
                100000000 * current,
                100 * eff_dist,
                100 * pdn_density,
                100000 * ir_drop_map
            ], out_file_name)
    else:
        print('Warning! Folder not found: {}. It will not be used for training!'.format(INPUT_PATH + "began_processed/"))

    if os.path.isdir(INPUT_PATH + "designs_Telpukhov/fake/"):
        files = glob.glob(INPUT_PATH + "designs_Telpukhov/fake/*/current_map.csv")
        print("Fake files found:", len(files))
        for f in files:
            id = os.path.basename(os.path.dirname(f))
            print(f, "ID", id)
            current = pd.read_csv(f, header=None).values
            eff_dist = pd.read_csv(os.path.dirname(f) + '/eff_dist_map.csv', header=None).values
            pdn_density = pd.read_csv(os.path.dirname(f) + '/pdn_density.csv', header=None).values
            ir_drop_map = pd.read_csv(os.path.dirname(f) + '/ir_drop_map.csv', header=None).values
            print(current.shape, eff_dist.shape, pdn_density.shape, ir_drop_map.shape)

            out_file_name = out_dir + 'fake_type_2_{}'.format(id) + '.pkl'
            save_in_file_fast([
                100000000 * current,
                100 * eff_dist,
                100 * pdn_density,
                100000 * ir_drop_map
            ], out_file_name)
    else:
        print('Warning! Folder not found: {}. It will not be used for training!'.format(INPUT_PATH + "designs_Telpukhov/fake/"))

    if os.path.isdir(INPUT_PATH + "ilya-contest-real-aug/"):
        files = glob.glob(INPUT_PATH + "ilya-contest-real-aug/*/*/current_map.csv")
        print("Fake files found:", len(files))
        for f in files:
            id = os.path.basename(os.path.dirname(f)) + '-' + os.path.basename(os.path.dirname(os.path.dirname(f)))
            print(f, "ID", id)
            current = pd.read_csv(f, header=None).values
            eff_dist = pd.read_csv(os.path.dirname(f) + '/eff_dist_map.csv', header=None).values
            pdn_density = pd.read_csv(os.path.dirname(f) + '/pdn_density.csv', header=None).values
            ir_drop_map = pd.read_csv(os.path.dirname(f) + '/ir_drop_map.csv', header=None).values
            print(current.shape, eff_dist.shape, pdn_density.shape, ir_drop_map.shape)

            out_file_name = out_dir + 'real_type_4_{}'.format(id) + '.pkl'
            save_in_file_fast([
                100000000 * current,
                100 * eff_dist,
                100 * pdn_density,
                100000 * ir_drop_map
            ], out_file_name)
    else:
        print('Warning! Folder not found: {}. It will not be used for training!'.format(INPUT_PATH + "ilya-contest-real-aug/"))

    if os.path.isdir(INPUT_PATH + "designs_Telpukhov/real/"):
        files = glob.glob(INPUT_PATH + "designs_Telpukhov/real/*/current_map.csv")
        print("Real files found:", len(files))
        for f in files:
            file_name = os.path.basename(f)
            id = os.path.basename(os.path.dirname(f))
            print(file_name, "ID", id)
            current = pd.read_csv(f, header=None).values
            eff_dist = pd.read_csv(os.path.dirname(f) + '/eff_dist_map.csv', header=None).values
            pdn_density = pd.read_csv(os.path.dirname(f) + '/pdn_density.csv', header=None).values
            ir_drop_map = pd.read_csv(os.path.dirname(f) + '/ir_drop_map.csv', header=None).values
            print(current.shape, eff_dist.shape, pdn_density.shape, ir_drop_map.shape)

            out_file_name = out_dir + 'real_type_3_{}'.format(id) + '.pkl'
            save_in_file_fast([
                100000000 * current,
                100 * eff_dist,
                100 * pdn_density,
                100000 * ir_drop_map
            ], out_file_name)
    else:
        print('Warning! Folder not found: {}. It will not be used for training!'.format(INPUT_PATH + "designs_Telpukhov/real/"))

    if os.path.isdir(INPUT_PATH + "designs_evgeniy/"):
        files = glob.glob(INPUT_PATH + "designs_evgeniy/DATA_*/*/current_map.csv")
        print("Fake files found:", len(files))
        for f in files:
            id = os.path.basename(os.path.dirname(f))
            print(f, "ID", id)
            current = pd.read_csv(f, header=None).values
            eff_dist = pd.read_csv(os.path.dirname(f) + '/eff_dist_map.csv', header=None).values
            pdn_density = pd.read_csv(os.path.dirname(f) + '/pdn_density.csv', header=None).values
            ir_drop_map = pd.read_csv(os.path.dirname(f) + '/ir_drop_map.csv', header=None).values
            print(current.shape, eff_dist.shape, pdn_density.shape, ir_drop_map.shape)
            if not (current.shape == eff_dist.shape == pdn_density.shape == ir_drop_map.shape):
                print('Not equal!')
                ir_drop_map = expand_matrix(ir_drop_map, current.shape[0], current.shape[1])

            out_file_name = out_dir + 'other_type_1_{}'.format(id) + '.pkl'
            save_in_file_fast([
                100000000 * current,
                100 * eff_dist,
                100 * pdn_density,
                100000 * ir_drop_map
            ], out_file_name)
    else:
        print('Warning! Folder not found: {}. It will not be used for training!'.format(INPUT_PATH + "designs_evgeniy/"))


if __name__ == '__main__':
    out_dir = CACHE_PATH + 'all_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    create_cache_file(out_dir)

