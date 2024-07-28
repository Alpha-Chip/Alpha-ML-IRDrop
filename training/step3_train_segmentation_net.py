# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'


if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import glob
import pickle
import torch
import time
import random
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.optim.lr_scheduler import CyclicLR

try:
    from problem_dataset_and_model import SegmDataset, get_train_aug, get_train_aug_large, get_valid_aug, get_model, MSELossUpdated
except:
    from .problem_dataset_and_model import SegmDataset, get_train_aug, get_train_aug_large, get_valid_aug, get_model, MSELossUpdated
try:
    from settings import *
except:
    from .settings import *


from torch.utils.data import DataLoader
# We need local version for train procedure which were removed in late version of module
import segmentation_models_pytorch_local as smp
import matplotlib.pyplot as plt


if __name__ == '__main__':
    FOLD_LIST = [0, 1, 2, 3, 4]
    print('Fold list: {}'.format(FOLD_LIST))


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def calculate_f1_score(actual_matrix, predicted_matrix):
    # Convert input matrices to NumPy arrays for easier element-wise operations
    predicted = np.array(predicted_matrix)
    actual = np.array(actual_matrix)

    # Hotspot matrices
    max_value = np.max(actual)
    threshold_value = 0.9 * max_value
    predicted = np.where(predicted > threshold_value, 1, 0)
    actual = np.where(actual > threshold_value, 1, 0)


    # Calculate True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)
    TP = np.sum(np.logical_and(predicted == 1, actual == 1))
    FP = np.sum(np.logical_and(predicted == 1, actual == 0))
    TN = np.sum(np.logical_and(predicted == 0, actual == 0))
    FN = np.sum(np.logical_and(predicted == 0, actual == 1))

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    # Calculate F1 score
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1_score


def valid(model, valid_data, epoch_num):
    model.eval()
    # with torch.cuda.amp.autocast():
    with torch.no_grad():
        needed_ids = sorted(list(set(valid_data['id'].unique())))

        all_data = []
        cache_files = glob.glob(CACHE_PATH + 'all_files/*.pkl')
        for f in cache_files:
            if 'real_type_3_' in os.path.basename(f):
                id = os.path.basename(f)[:-4].split('_')[-1]
                if id in needed_ids:
                    all_data.append(f)
            if 'real_type_4_' in os.path.basename(f):
                id = os.path.basename(f).split('-')[-2]
                if id in needed_ids:
                    all_data.append(f)

        # print(len(all_data))

        all_images = []
        all_masks = []
        all_ids = []
        for id in all_data:
            # print(el)
            arr = load_from_file_fast(id)
            img = np.array(arr[:3], dtype=np.float32)
            mask = np.array(arr[-1:], dtype=np.float32)
            # print(img.shape, mask.shape)
            all_images.append(img)
            all_masks.append(mask)
            all_ids.append(os.path.basename(id))

        all_diff = 0
        all_square_diff = 0
        all_counts = 0
        f1_score_list = []

        for i in range(len(all_images)):
            image = all_images[i]
            image_shape = image.shape
            mask = all_masks[i]
            id = all_ids[i]

            # Fix small images
            if image.shape[1] < SHAPE_SIZE[1] or image.shape[2] < SHAPE_SIZE[2]:
                image_new = np.zeros((
                    image.shape[0],
                    max(SHAPE_SIZE[1], image.shape[1]),
                    max(SHAPE_SIZE[2], image.shape[2]),
                ), dtype=np.float32)
                mask_new = np.zeros((
                    mask.shape[0],
                    max(SHAPE_SIZE[1], mask.shape[1]),
                    max(SHAPE_SIZE[2], mask.shape[2]),
                ), dtype=np.float32)
                image_new[:, :image.shape[1], :image.shape[2]] = image
                mask_new[:, :mask.shape[1], :mask.shape[2]] = mask
                image = image_new
                mask = mask_new

            step = SHAPE_SIZE[1] // 2
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
                    steps_count += 1

                    sub_image = torch.unsqueeze(torch.from_numpy(image[:, start_1:end_1, start_2:end_2]), dim=0).cuda()
                    pmask = model(sub_image)[0]
                    pred_mask[:, start_1:end_1, start_2:end_2] += pmask.cpu().numpy()
                    pred_count[:, start_1:end_1, start_2:end_2] += 1.

            # Fix small images (remove padding from zeros)
            pred_mask = pred_mask[:, :image_shape[1], :image_shape[2]]
            pred_count = pred_count[:, :image_shape[1], :image_shape[2]]
            mask = mask[:, :image_shape[1], :image_shape[2]]

            pred_mask /= pred_count
            all_diff += np.abs(pred_mask - mask).sum()
            all_square_diff += ((pred_mask - mask) * (pred_mask - mask)).sum()
            all_counts += len(mask.flatten())

            if 1:
                mae_value = np.abs(pred_mask - mask).sum() / len(mask.flatten())
                f1_score = calculate_f1_score(mask, pred_mask)
                f1_score_list.append(f1_score)

                if 'real_type_3_' in id:
                    print("ID: {} F1 Score: {:.4f} MAE score: {:.4f} Steps: {}".format(id, f1_score, mae_value, steps_count))

                fig = plt.figure(figsize=(16, 8))
                plt.subplot(1, 2, 1)
                plt.title('Real')
                plt.imshow(mask[0], interpolation='none')
                plt.colorbar()

                plt.subplot(1, 2, 2)
                plt.title('Pred [MAE: {:.4f}]'.format(mae_value))
                plt.imshow(pred_mask[0], interpolation='none')
                plt.colorbar()

                plt.savefig(CACHE_PATH + 'debug_mae_{:.4f}_ep_{}_test_{}.png'.format(mae_value, epoch_num, id))
                plt.close(fig)
                # plt.show()

        score_mae = all_diff / all_counts
        score_mse = all_square_diff / all_counts
        score_f1 = np.array(f1_score_list).mean()

    return score_mae, score_mse, score_f1


def train_single_model(fold_number, start_weights):

    print('Go fold: {} Input size: {}'.format(fold_number, INPUT_SIZE))
    cnn_type = '{}_{}_drop_{}_size_{}'.format(BACKBONE, DECODER_TYPE, DROPOUT_VALUE, SHAPE_SIZE[-1])
    print('Creating and compiling {}...'.format(cnn_type))

    split = pd.read_csv(KFOLD_SPLIT)
    train_data = split[split['fold'] != fold_number]
    valid_data = split[split['fold'] == fold_number]

    model = get_model(BACKBONE, DECODER_TYPE, dropout=DROPOUT_VALUE)
    if USE_PARALLEL:
        if torch.cuda.device_count() > 1:
            print("Running on {} GPUS!".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
    # model.to(DEVICE)
    # preprocessing_fn = get_preprocessing_fn_v2(BACKBONE, pretrained='imagenet')
    # print(summary(model, input_size=SHAPE_SIZE))

    if start_weights:
        print('Start from weights: {}'.format(start_weights))
        model.load_state_dict(torch.load(start_weights))

    train_dataset = SegmDataset(
        train_data,
        is_valid=False,
        augmentation=get_train_aug(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=THREADS
    )

    # loss = smp.utils.losses.JaccardLoss(activation='sigmoid') + smp.utils.losses.BCEWithLogitsLoss()
    # loss = smp.utils.losses.JaccardLoss(activation='sigmoid')
    if USE_UPDATED_MSE_LOSS:
        print('Use special MSE loss beta = 2')
        loss = MSELossUpdated(beta=2)
    else:
        print('Use standard MSE loss')
        loss = smp.utils.losses.MSELoss()

    optimizer = torch.optim.Adam([
        dict(
            params=model.parameters(),
            lr=LEARNING_RATE
        ),
    ])

    if USE_REDUCE_ON_PLATEU:
        scheduler_ReduceLROnPlateau = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.9,
            patience=4,
            verbose=True,
        )
    else:
        scheduler_StepLR = StepLR(
            optimizer,
            step_size=5,
            gamma=0.1,
        )

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=[],
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    score_mae, score_mse, score_f1 = valid(model, valid_data, "init")
    print('Score MAE: {:.6f} Score MSE: {:.6f} Score F1: {:.6f}'.format(score_mae, score_mse, score_f1))

    min_score = 10000000000
    last_model_path = MODELS_PATH_TORCH + './model_{}_fold_{}_last_model.pth'.format(cnn_type, fold_number)
    no_improvements = 0
    for ep in range(0, EPOCHS):
        if no_improvements > PATIENCE:
            print('Epochs without improvememnts: {}. Break training...'.format(no_improvements))
            break
        print('\nEpoch: {}/{} Learning rate: {}'.format(ep, EPOCHS, optimizer.param_groups[0]['lr']))
        train_logs = train_epoch.run(train_loader)
        valid_time = time.time()
        score_mae, score_mse, score_f1 = valid(model, valid_data, ep)
        print('Valid size: {} Score MAE: {:.6f} Score MSE: {:.6f} Score F1: {:.6f} Time: {:.2f} sec'.format(len(valid_data), score_mae, score_mse, score_f1, time.time() - valid_time))
        # valid_logs = valid_epoch.run(valid_loader)
        # print(valid_logs)

        # do something (save model, change lr, etc.)
        if min_score > score_mae:
            no_improvements = 0
            min_score = score_mae
            print('Best model saved for MAE: {}!'.format(score_mae))
            save_path = MODELS_PATH_TORCH + './model_{}_fold_{}_mae_{:.6f}_f1_{:.6f}_ep_{:02d}.pth'.format(cnn_type, fold_number, score_mae, score_f1, ep)
            print('Saved: {}'.format(os.path.basename(save_path)))
            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()
            torch.save(state_dict, save_path)

        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, last_model_path)

        if USE_REDUCE_ON_PLATEU:
            scheduler_ReduceLROnPlateau.step(score_mae)
        else:
            scheduler_StepLR.step(ep)
        no_improvements += 1

    return min_score, save_path


if __name__ == '__main__':
    start_time = time.time()
    random.seed(start_time)
    np.random.seed(int(start_time))

    # if you trained models before you can start from previous weights
    start_weights = [
        MODELS_PATH_TORCH + 'model_tu-maxvit_large_tf_512_Unet_drop_0.0_size_512_fold_0_last_model.pth',
        None,
        None,
        None,
        None,
    ]

    for kf in range(KFOLD_NUMBER):
        if kf not in FOLD_LIST:
            print('Skip fold:', kf)
            continue
        train_single_model(kf, start_weights[kf])
    print('Time: {:.0f} sec'.format(time.time() - start_time))


'''

'''