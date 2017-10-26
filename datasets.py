# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:06:25 2017

@author: sakurai
"""

import cv2
import numpy as np
from chainer.datasets import get_mnist, get_cifar10, get_cifar100
from chainer.dataset import concat_examples


def load_mnist_as_ndarray(ndim):
    train, test = get_mnist(ndim=ndim)
    train = concat_examples(train)
    test = concat_examples(test)
    return train, test


def load_cifar10_as_ndarray(ndim):
    train, test = get_cifar10(ndim=ndim)
    train = concat_examples(train)
    test = concat_examples(test)
    return train, test


def load_cifar100_as_ndarray(ndim):
    train, test = get_cifar100(ndim=ndim)
    train = concat_examples(train)
    test = concat_examples(test)
    return train, test


def random_flip_lr(x):
    num_examples = len(x)
    for i, flip in enumerate(np.random.randint(0, 2, num_examples)):
        if flip:
            x[i] = x[i, :, :, ::-1]
    return x


def random_augment_scaling(bchw, max_expand=3):
    '''Data augmentation by random expanding by scaling up and cropping
    '''
    if isinstance(max_expand, int):
        max_expand = (max_expand, max_expand)
    max_expand_y, max_expand_x = max_expand

    bhwc = bchw.transpose(0, 2, 3, 1)
    batch_size, height, width = bhwc.shape[:3]

    # opencv treats image coordinate as (x, y)
    expands = np.vstack((np.random.randint(0, max_expand_x + 1, batch_size),
                         np.random.randint(0, max_expand_y + 1, batch_size))).T
    sizes = np.array([height, width]) + expands
    results = np.empty_like(bhwc)
    for hwc, size, expand, result in zip(bhwc, sizes, expands, results):
        resized = cv2.resize(hwc, tuple(size))
        expand_x, expand_y = expand
        ox = np.random.randint(0, expand_x + 1)
        oy = np.random.randint(0, expand_y + 1)
        cropped = resized[oy:height+oy, ox:width+ox]
        if np.random.choice(2):
            cropped = cropped[:, ::-1]
        result[:] = cropped

    return results.transpose(0, 3, 1, 2)


def random_augment_padding(bchw, pad=4):
    '''Data augmentation by random expanding by padding and cropping
    '''
    if isinstance(pad, int):
        pad = (pad, pad)
    pad_y, pad_x = pad

    height, width = bchw.shape[2:]
    padded_bchw = np.pad(
        bchw, ((0, 0), (0, 0), (pad_y, pad_y), (pad_x, pad_x)), 'constant')

    results = np.empty_like(bchw)
    for padded_chw, result in zip(padded_bchw, results):
        ox = np.random.randint(0, 2 * pad_x + 1)
        oy = np.random.randint(0, 2 * pad_y + 1)
        cropped = padded_chw[:, oy:height+oy, ox:width+ox]
        if np.random.choice(2):
            cropped = cropped[:, :, ::-1]
        result[:] = cropped

    return results
