# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'

import os
import time
import glob
import pandas as pd

from utils.converter import convert_spice, convert_irdrop_to_csv_file, expand_matrix
import multiprocessing
import gzip

INPUT_PATH = '../input/'

def proc_single_dir(file_name):
    print('Go: {}'.format(file_name))
    base_name = os.path.basename(file_name)[:-6]
    set_number = os.path.basename(os.path.dirname(os.path.dirname(file_name)))
    output_folder = INPUT_PATH + 'began_processed/' + set_number + '_' + base_name
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # Ungzip spice
    f = gzip.open(file_name, 'rb')
    try:
        file_content = f.read()
    except:
        print('Problem with: {}'.format(file_name))
        return
    out = open(output_folder + '/netlist.sp', 'wb')
    out.write(file_content)

    # Ungzip Irdrop
    irdrop_file_name = os.path.dirname(file_name) + '/' + base_name + '_ir_drop.csv.gz'
    f = gzip.open(irdrop_file_name, 'rb')
    file_content = f.read()
    out = open(output_folder + '/ir_drop_map.csv', 'wb')
    out.write(file_content)
    irdrop_read = pd.read_csv(output_folder + '/ir_drop_map.csv', header=None)

    current_map_file = os.path.join(output_folder, "current_map.csv")
    voltage_map_file = os.path.join(output_folder, "eff_dist_map.csv")
    density_map_file = os.path.join(output_folder, "pdn_density.csv")

    start_time = time.time()
    convert_spice(
        output_folder + '/netlist.sp',
        current_map_file,
        voltage_map_file,
        density_map_file,
        force_dim=irdrop_read.values.shape
    )
    print('Completed in: {:.2f} sec'.format(time.time() - start_time))


def convert_data_to_contest_format(began_data):
    files = glob.glob(began_data + 'nangate45/*/data/*.sp.gz')
    print("Found files: {}".format(len(files)))

    if 0:
        for f in files:
            proc_single_dir(f)

    pool = multiprocessing.Pool(processes=6)
    pool.map(proc_single_dir, files)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # Download BeGAN from: https://github.com/UMN-EDA/BeGAN-benchmarks/tree/master/BeGAN-circuit-benchmarks
    began_data = INPUT_PATH + 'BeGAN/'
    convert_data_to_contest_format(began_data)


