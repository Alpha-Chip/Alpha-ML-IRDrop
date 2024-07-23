# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'


try:
    from settings import *
except:
    from .settings import *

from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import pandas as pd
import tqdm
import glob
import pickle
import random
import torch.nn as nn
import torch
import re
import segmentation_models_pytorch as smp

INPUT_SIZE = SHAPE_SIZE[-1]


def show_image(im, name='image'):
    import cv2
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


class Loss(BaseObject):

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)


class SumOfLosses(Loss):

    def __init__(self, l1, l2):
        name = '{} + {}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)

    def forward(self, *inputs):
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


class MultipliedLoss(Loss):

    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split('+')) > 1:
            name = '{} * ({})'.format(multiplier, loss.__name__)
        else:
            name = '{} * {}'.format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs):
        return self.multiplier * self.loss.forward(*inputs)


class MSELossUpdated(Loss):

    def __init__(self, beta=3, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def forward(self, y_pr, y_gt):
        diff_pos = (y_pr > y_gt).type(y_pr.dtype)
        diff_neg = (y_pr <= y_gt).type(y_pr.dtype)
        part_pos = (y_pr - y_gt) * (y_pr - y_gt) * diff_pos
        part_neg = (y_pr - y_gt) * (y_pr - y_gt) * diff_neg
        part_pos_mse = part_pos.sum() / (diff_pos.sum() + 1e-7)
        part_neg_mse = part_neg.sum() / (diff_neg.sum() + 1e-7)
        return part_pos_mse + self.beta * part_neg_mse


def get_train_aug():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
        ],
        is_check_shapes=False,
        p=1.0
    )


def get_train_aug_large():
    return A.Compose(
        [
            A.ShiftScaleRotate(p=0.5, shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT),
            A.RandomCropFromBorders(p=0.5, crop_value=0.1),
            A.Resize(height=INPUT_SIZE, width=INPUT_SIZE, p=1),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.05 * 255), p=1.0),
                A.GaussNoise(var_limit=(1.0, 5.0), p=1.0),
            ], p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
        ],
        is_check_shapes=False,
        p=1.0
    )

def get_valid_aug():
    return A.Compose(
        [
            A.Resize(height=INPUT_SIZE, width=INPUT_SIZE, p=0.0),
        ],
        p=0.0,
    )


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


class SegmDataset(Dataset):

    def __init__(
            self,
            dataframe,
            is_valid=False,
            augmentation=None,
            preprocessing=None,
    ):
        super().__init__()
        self.all_data = []
        cache_files = glob.glob(CACHE_PATH + 'all_files/*.pkl')
        needed_ids = set(dataframe['id'].unique())

        for f in cache_files:
            if 'real_type_3_' in os.path.basename(f):
                continue
            if 'real_type_4_' in os.path.basename(f):
                id = os.path.basename(f).split('-')[-2]
                if id not in needed_ids:
                    continue
                else:
                    # print('Add {}'.format(f))
                    a = 1
            self.all_data.append(f)

        if not is_valid:
            print('Train dataset size: {}'.format(len(self.all_data)))
        else:
            print('Valid dataset size: {}'.format(len(self.all_data)))

        self.df = dataframe
        self.is_valid = is_valid
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        index = random.randint(0, len(self.all_data)-1)
        arr = load_from_file_fast(self.all_data[index])
        image = np.array(arr[:3], dtype=np.float32)
        mask = np.array(arr[-1], dtype=np.float32)
        image = np.transpose(image, (1, 2, 0))

        # apply augmentations
        if self.augmentation:
            # print(self.all_data[index])
            # print(image.shape, image.dtype, image.mean(), image.min(), image.max())
            # print(mask.shape, mask.dtype, mask.mean(), mask.min(), mask.max())
            if image.shape[:-1] != mask.shape and 0:
                print('Some problem!', image.shape[:-1], mask.shape, self.all_data[index])
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            # print(image.shape, image.mean(), image.min(), image.max())
            # print(mask.shape, mask.mean(), mask.min(), mask.max())

        mask = np.expand_dims(mask, axis=0)
        image = np.transpose(image, (2, 0, 1))

        if 0:
            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

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
            # print("Image: {} Mask: {}".format(image.shape, mask.shape))

        # extract random part
        start_1 = random.randint(0, image.shape[1] - SHAPE_SIZE[1])
        end_1 = start_1 + SHAPE_SIZE[1]
        start_2 = random.randint(0, image.shape[2] - SHAPE_SIZE[2])
        end_2 = start_2 + SHAPE_SIZE[2]

        image = image[:, start_1:end_1, start_2:end_2]
        mask = mask[:, start_1:end_1, start_2:end_2]

        return image, mask

    def __len__(self):
        return ITERATIONS_PER_EPOCH

    def get_all_items(self):
        return self.all_images.copy()


def get_model(backbone, decoder_type, dropout=0.0):
    in_channels = 3
    classes = CLASSES_NUMBER

    if decoder_type == 'Unet':
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            # encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            # activation='irdrop',
        )
    elif decoder_type == 'FPN':
        model = smp.FPN(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            decoder_dropout=dropout,
        )

    return model


def normalize_array(cube, new_max, new_min):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(cube), np.max(cube)
    if maximum - minimum != 0:
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        cube = m * cube + b
    return cube


def show_images_for_check(image, mask, is_valid):
    image[0, :, :] = normalize_array(image[0, :, :], 255, 0)
    image[1, :, :] = normalize_array(image[1, :, :], 255, 0)
    image[2, :, :] = normalize_array(image[2, :, :], 255, 0)
    show_image(image.transpose((1, 2, 0)))


if __name__ == '__main__':
    split = pd.read_csv(KFOLD_SPLIT)
    train_data = split[split['fold'] != 0]
    valid_data = split[split['fold'] == 0]
    print(len(train_data), len(valid_data))
    train_dataset = SegmDataset(
        train_data,
        is_valid=False,
        augmentation=get_train_aug(),
    )
    for image, mask in train_dataset:
        print(image.shape, mask.shape)
        print(image.min(), image.max(), image.mean())
        print(mask.min(), mask.max(), mask.mean())
        show_images_for_check(image, mask, is_valid=False)


