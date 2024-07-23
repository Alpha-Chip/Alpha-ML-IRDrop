# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'


if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import sys
import io
import torch
import time
import pandas as pd
import numpy as np
import pickle
import gzip
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import segmentation_models_pytorch as smp


SHAPE_SIZE = (3, 512, 512)
CLASSES_NUMBER = 1


def get_model(backbone, decoder_type, dropout=0.0):
    in_channels = 3
    classes = CLASSES_NUMBER

    if decoder_type == 'Unet':
        model = smp.Unet(
            encoder_name=backbone,
            # encoder_weights="imagenet",
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )
    elif decoder_type == 'FPN':
        model = smp.FPN(
            encoder_name=backbone,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            decoder_dropout=dropout,
        )

    return model


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def get_masks_tta8(data, model):
    data0 = data
    data1 = torch.rot90(data, 1, (2, 3))
    data2 = torch.rot90(data, 2, (2, 3))
    data3 = torch.rot90(data, 3, (2, 3))
    data4 = torch.flip(data, [2])
    data5 = torch.rot90(data4, 1, (2, 3))
    data6 = torch.rot90(data4, 2, (2, 3))
    data7 = torch.rot90(data4, 3, (2, 3))

    pred_masks = []
    for i, d in enumerate([data0, data1, data2, data3, data4, data5, data6, data7]):
        pred_mask = model(d)
        pred_masks.append(pred_mask)
    pred_masks = torch.stack(pred_masks, 0)

    pred_masks[0] = pred_masks[0]
    pred_masks[1] = torch.rot90(pred_masks[1], -1, (2, 3))
    pred_masks[2] = torch.rot90(pred_masks[2], -2, (2, 3))
    pred_masks[3] = torch.rot90(pred_masks[3], -3, (2, 3))
    pred_masks[4] = torch.flip(pred_masks[4], [2])
    pred_masks[5] = torch.rot90(pred_masks[5], -1, (2, 3))
    pred_masks[5] = torch.flip(pred_masks[5], [2])
    pred_masks[6] = torch.rot90(pred_masks[6], -2, (2, 3))
    pred_masks[6] = torch.flip(pred_masks[6], [2])
    pred_masks[7] = torch.rot90(pred_masks[7], -3, (2, 3))
    pred_masks[7] = torch.flip(pred_masks[7], [2])

    pred_masks = pred_masks.cpu().numpy()
    pred_mask = np.array(pred_masks).mean(axis=0)
    # We need to apply sigmoid first, because we have only logits on outputs!
    # pred_mask = sigmoid(pred_mask)
    return pred_mask


def compute_intersection_area(x, y, i, j):
    # Координаты левого верхнего угла квадрата (i, j)
    x1, y1 = i - 0.5, j - 0.5
    # Координаты правого нижнего угла квадрата (i+1, j+1)
    x2, y2 = i + 0.5, j + 0.5

    # Вычисление координаты левого верхнего угла пересечения
    x_left = max(x - 0.5, x1)
    y_top = min(y + 0.5, y2)

    # Вычисление координаты правого нижнего угла пересечения
    x_right = min(x + 0.5, x2)
    y_bottom = max(y - 0.5, y1)

    # Вычисление площади пересечения
    intersection_area = max(0, x_right - x_left) * max(0, y_top - y_bottom)

    return intersection_area


def expand_matrix(matrix, n1, m1):
    n, m = matrix.shape
    result_matrix = np.zeros((n1, m1))

    nrows = min(n, n1)
    ncols = min(m, m1)

    result_matrix[:nrows, :ncols] = matrix[:nrows, :ncols]

    if m1 > m:
        # Расширение матрицы вправо
        result_matrix[:nrows, m:] = np.repeat(matrix[:nrows, -1][:, np.newaxis], m1 - m, axis=1)

    if n1 > n:
        # Расширение матрицы вниз
        result_matrix[n:, :] = np.repeat(result_matrix[nrows - 1, :][np.newaxis, :], n1 - nrows, axis=0)

    return result_matrix


def shift(x, y):
    # Создание нового графика
    x -= 0.5
    y -= 0.5

    # Вычисление и вывод площади пересечения с каждым из 9 квадратов
    matrix = np.zeros((3, 3))
    for i in range(-1, 2):
        for j in range(-1, 2):
            intersection_area = compute_intersection_area(x, y, i, j)
            matrix[i + 1][j + 1] = intersection_area
    # Показать график
    return matrix


def find_core(sp):
    min_x = min(min(sp["I"]["x"]), min(sp["V"]["x"]), min(sp["R"]["x1"]), min(sp["R"]["x2"]))
    max_x = max(max(sp["I"]["x"]), max(sp["V"]["x"]), max(sp["R"]["x1"]), max(sp["R"]["x2"]))
    min_y = min(min(sp["I"]["y"]), min(sp["V"]["y"]), min(sp["R"]["y1"]), min(sp["R"]["y2"]))
    max_y = max(max(sp["I"]["y"]), max(sp["V"]["y"]), max(sp["R"]["y1"]), max(sp["R"]["y2"]))
    return (min_x, max_x, min_y, max_y)


def read_sp(filename):
    # вычисляем координаты всех инстансов
    sp = {
        "I": {
            "x": [],
            "y": [],
            "v": [],
            "m": []
        },
        "V": {
            "x": [],
            "y": [],
            "v": [],
            "m": []
        },
        "R": {
            "x1": [],
            "y1": [],
            "x2": [],
            "y2": [],
            "m1": [],
            "m2": [],
            "v": []
        }
    }
    with open(filename, 'r') as file:
        for line in file:
            if line[0] == "I":
                parts = line.strip().split(' ')
                coordinates = parts[1].split('_')
                sp["I"]["x"].append(int(coordinates[2]))
                sp["I"]["y"].append(int(coordinates[3]))
                sp["I"]["v"].append(float(parts[3]))
                sp["I"]["m"].append(coordinates[1])
            elif line[0] == "R":
                parts = line.strip().split(' ')
                coordinates1 = parts[1].split('_')
                coordinates2 = parts[2].split('_')
                sp["R"]["x1"].append(int(coordinates1[2]))
                sp["R"]["y1"].append(int(coordinates1[3]))
                sp["R"]["x2"].append(int(coordinates2[2]))
                sp["R"]["y2"].append(int(coordinates2[3]))
                sp["R"]["m1"].append(coordinates1[1])
                sp["R"]["m2"].append(coordinates2[1])
                sp["R"]["v"].append(float(parts[3]))
            elif line[0] == "V":
                parts = line.strip().split(' ')
                coordinates = parts[1].split('_')
                sp["V"]["x"].append(int(coordinates[2]))
                sp["V"]["y"].append(int(coordinates[3]))
                sp["V"]["v"].append(float(parts[3]))
                sp["V"]["m"].append(coordinates[1])
    return sp


def netlist2currmap(sp, force_dim=None, step_x=2000, step_y=2000, mv=(0, 0)):
    # Создание пустой матрицы
    x_coord = sp["I"]["x"]
    y_coord = sp["I"]["y"]
    values = sp["I"]["v"]
    # Вычисляем размер ядра
    _, max_x, _, max_y = find_core(sp)
    x_width = int(max_x / step_x) + 1
    y_width = int(max_y / step_y) + 1
    matrix = np.zeros((x_width, y_width))
    for i in range(len(values)):
        posx = (x_coord[i] // step_x)
        posy = (y_coord[i] // step_y)

        # Вычисление коэффициентов для распределения значения
        dx = x_coord[i] / step_x - posx
        dy = y_coord[i] / step_y - posy
        sh = shift(dx, dy)

        # Определение, лежит ли точка на границе
        is_bndry = 0
        if posx in [0, x_width - 1]:
            is_bndry = 1
        if posy in [0, y_width - 1]:
            is_bndry = 1

        # Распределение значения между ячейками
        if is_bndry:
            matrix[posx, posy] += values[i]
        else:
            matrix[posx - 1:posx - 1 + 3, posy - 1:posy - 1 + 3] += sh * values[i]
    matrix = gaussian_filter(matrix, sigma=0.6)


    if force_dim != None:
        matrix = expand_matrix(matrix, force_dim[0], force_dim[1])
    return matrix[mv[0]:,mv[1]:]


def netlist2voltge(sp, force_dim=None, step_x=2000, step_y=2000, mv=(0, 0)):
    # Создание пустой матрицы
    x_coord = sp["V"]["x"]
    y_coord = sp["V"]["y"]
    values = sp["V"]["v"]
    # Вычисляем размер ядра
    _, max_x, _, max_y = find_core(sp)
    x_width = int(max_x / step_x) + 1
    y_width = int(max_y / step_y) + 1
    matrix = np.zeros((x_width, y_width))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            x = 1000 + 2000 * i
            y = 1000 + 2000 * j
            dist = []
            sum_1 = 0
            for k in range(len(values)):
                x_p = x_coord[k]
                y_p = y_coord[k]
                dst = ((x_p - x) ** 2 + (y_p - y) ** 2) ** (0.5)
                dist.append(dst)
                sum_1 += 1 / dst
            matrix[i, j] = (1 / sum_1) / 2000

    if force_dim != None:
        matrix = expand_matrix(matrix, force_dim[0], force_dim[1])
    return matrix[mv[0]:,mv[1]:]


def netlist2density(sp, force_dim=None, step_x=2000, step_y=2000, mv=(0, 0)):
    # Создание пустой матрицы
    x1 = sp["R"]["x1"]
    y1 = sp["R"]["y1"]
    x2 = sp["R"]["x2"]
    y2 = sp["R"]["y2"]
    values = sp["R"]["v"]
    # Вычисляем размер ядра
    _, max_x, _, max_y = find_core(sp)
    x_width = int(max_x / step_x) + 1
    y_width = int(max_y / step_y) + 1
    matrix = np.zeros((x_width, y_width))
    for i in range(len(values)):
        posx = int(x1[i] // step_x)
        posy = int(y1[i] // step_y)
        matrix[posx, posy] += values[i]
        posx = int(x2[i] // step_x)
        posy = int(y2[i] // step_y)
        matrix[posx, posy] += values[i]
    matrix = gaussian_filter(matrix, 4)
    if force_dim != None:
        matrix = expand_matrix(matrix, force_dim[0], force_dim[1])
    return matrix[mv[0]:,mv[1]:]


def convert_spice(sp_in, force_dim=None):
    sp = read_sp(sp_in)
    curr_map = netlist2currmap(sp, force_dim=force_dim)
    voltage_map = netlist2voltge(sp, force_dim=force_dim)
    density_map = netlist2density(sp, force_dim=force_dim)
    return curr_map, voltage_map, density_map


USE_TTA = False
STEP_DIV = 2
CORRECTION_COEFF = 1.05


def run_all(
    current_map,
    pdn_density,
    eff_dist_map,
    netlist,
    checkpoint,
    output_file,
):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('Use device: {}'.format(device))

    print('Start from weights: {}'.format(checkpoint))
    weigts_list = load_from_file_fast(checkpoint)
    models_list = []
    for weigth in weigts_list:
        model = get_model('tu-maxvit_large_tf_512', 'Unet', dropout=0.0)
        model.load_state_dict(weigth)
        model.to(device)
        model.eval()
        models_list.append(model)

    with torch.no_grad():
        current = pd.read_csv(current_map, header=None)
        current, eff_dist, pdn_density = convert_spice(netlist, current.values.shape)
        print(current.shape, eff_dist.shape, pdn_density.shape)
        arr = [
            100000000 * current,
            100 * eff_dist,
            100 * pdn_density,
        ]
        image = np.array(arr, dtype=np.float32)
        image_shape = image.shape
        print('Input image shape: {}'.format(image.shape))

        # Fix small images (add padding from zeros)
        if image.shape[1] < SHAPE_SIZE[1] or image.shape[2] < SHAPE_SIZE[2]:
            image_new = np.zeros((
                image.shape[0],
                max(SHAPE_SIZE[1], image.shape[1]),
                max(SHAPE_SIZE[2], image.shape[2]),
            ), dtype=np.float32)
            image_new[:, :image.shape[1], :image.shape[2]] = image
            image = image_new

        step = SHAPE_SIZE[1] // STEP_DIV
        pred_mask = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.float32)
        pred_count = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.float32)
        steps = dict()
        steps_count = 0
        for j in range(0, image.shape[1], step):
            for k in range(0, image.shape[2], step):
                start_1 = j
                end_1 = start_1 + SHAPE_SIZE[1]
                if end_1 > image.shape[1]:
                    end_1 = image.shape[1]
                    start_1 = end_1 - SHAPE_SIZE[1]

                start_2 = k
                end_2 = start_2 + SHAPE_SIZE[2]
                if end_2 > image.shape[2]:
                    end_2 = image.shape[2]
                    start_2 = end_2 - SHAPE_SIZE[2]

                # Skip if we try 2nd time
                if (start_1, start_2) in steps:
                    continue

                steps[(start_1, start_2)] = 1
                sub_image = torch.unsqueeze(torch.from_numpy(image[:, start_1:end_1, start_2:end_2]), dim=0).to(device)

                for model in models_list:
                    if USE_TTA:
                        pmask = get_masks_tta8(sub_image, model)[0]
                    else:
                        pmask = model(sub_image)[0].cpu().numpy()
                    pred_mask[:, start_1:end_1, start_2:end_2] += pmask
                    pred_count[:, start_1:end_1, start_2:end_2] += 1.
                    steps_count += 1

        # Fix small images (remove padding from zeros)
        pred_mask = pred_mask[:, :image_shape[1], :image_shape[2]]
        pred_count = pred_count[:, :image_shape[1], :image_shape[2]]
        pred_mask /= pred_count

        # Try to make F1 score non-zero
        thr_value = pred_mask.max() * 0.9
        pred_mask[pred_mask > thr_value] *= CORRECTION_COEFF

        # divide 100000
        pred_mask /= 100000

        # save results
        pred_mask = pred_mask[0]
        out = open(output_file, 'w')

        for i in range(pred_mask.shape[0]):
            out.write(str(pred_mask[i, 0]))
            for j in range(1, pred_mask.shape[1]):
                out.write(",{}".format(str(pred_mask[i, j])))
            out.write('\n')

        out.close()


if __name__ == '__main__':
    start_time = time.time()

    argv = sys.argv
    current_map = argv[1]
    pdn_density = argv[2]
    eff_dist_map = argv[3]
    netlist = argv[4]
    checkpoint = argv[5]
    output_file = argv[6]

    run_all(
        current_map,
        pdn_density,
        eff_dist_map,
        netlist,
        checkpoint,
        output_file,
    )
    print('Time: {:.0f} sec'.format(time.time() - start_time))
