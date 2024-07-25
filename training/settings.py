# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'

import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/'
INPUT_PATH = ROOT_PATH + 'input/'
OUTPUT_PATH = ROOT_PATH + 'output/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
MODELS_PATH = ROOT_PATH + 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
CACHE_PATH = ROOT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)

if 1:
    BATCH_SIZE = 6
    SHAPE_SIZE = (3, 512, 512)
    EPOCHS = 100
    USE_REDUCE_ON_PLATEU = True

USE_PARALLEL = False
THREADS = 1
INPUT_SIZE = SHAPE_SIZE[-1]
KFOLD_NUMBER = 5
USE_UPDATED_MSE_LOSS = False
BACKBONE = 'tu-maxvit_large_tf_512'
DECODER_TYPE = 'Unet'
DROPOUT_VALUE = 0.0
CLASSES_NUMBER = 1
PATIENCE = 30
DEVICE = 'cuda'
LEARNING_RATE = 0.0001
ITERATIONS_PER_EPOCH = BATCH_SIZE * 1000
KFOLD_SPLIT = OUTPUT_PATH + 'kfold_split_5_seed_42.csv'

DIR_PREFIX = os.path.basename(os.path.dirname(__file__))
MODELS_PATH_TORCH = MODELS_PATH + DIR_PREFIX + '_r/'
if not os.path.isdir(MODELS_PATH_TORCH):
    os.mkdir(MODELS_PATH_TORCH)
