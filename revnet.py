# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:37:15 2017

@author: sakurai

An implementation of RevNet for Chainer.
"The Reversible Residual Network: Backpropagation Without Storing Activations".
"""

from copy import deepcopy
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from datasets import load_cifar10_as_ndarray, random_augment_padding

from links import BRCChain


class Revnet(chainer.Chain):
    '''
    Reversible Residual Network.

    Args:
        n (int):
            Number of units in each group.
    '''

    def __init__(self, n=6):
        super(Revnet, self).__init__(
            conv1=L.Convolution2D(3, 32, 3, pad=1, nobias=True),
            stage2=RevnetStage(n, 32),
            stage3=RevnetStage(n, 64),
            stage4=RevnetStage(n, 128),
            bn_out=L.BatchNormalization(128),
            fc_out=L.Linear(128, 10)
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = self.stage2(h)
        h = F.max_pooling_2d(h, 2)
        h = self.stage3(h)
        h = F.max_pooling_2d(h, 2)
        h = self.stage4(h)
        h = self.bn_out(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, h.shape[2:])
        y = self.fc_out(h)
        return y


class RevnetStage(chainer.ChainList):
    '''Reversible sequence of `ResnetUnit`s.
    '''
    def __init__(self, n_blocks, channels):
        blocks = [ResnetUnit(channels // 2) for i in range(n_blocks)]
        super(RevnetStage, self).__init__(*blocks)
        self._channels = channels

    def __call__(self, x):
        x = extend_channels(x, self._channels)
        revnet_stage_function = RevnetStageFunction(self)
        y = revnet_stage_function(x)
        return y


class RevnetStageFunction(chainer.Function):
    def __init__(self, chainlist):
        """
        Args:
            chainlist (chainer.Chainlist):
                A ChainList of revnet units.
        """
        self.chainlist = chainlist

    def forward(self, inputs):
        xp = chainer.cuda.get_array_module(*inputs)
        x = inputs[0]
        x1, x2 = xp.split(x, 2, axis=1)

        with chainer.no_backprop_mode():
            x1 = chainer.Variable(x1)
            x2 = chainer.Variable(x2)
            for res_unit in self.chainlist:
                x2 += res_unit(x1)
                x1, x2 = x2, x1

        y = xp.concatenate((x1.array, x2.array), axis=1)
        self.retain_outputs((0,))
        return y,

    def backward(self, inputs, grads):
        xp = chainer.cuda.get_array_module(*grads)
        y_array = self.output_data[0]
        grad_y = grads[0]

        y1_array, y2_array = xp.split(y_array, 2, axis=1)
        grad_y1, grad_y2 = xp.split(grad_y, 2, axis=1)

        a, b = y1_array.copy(), y2_array.copy()
        ga, gb = grad_y1.copy(), grad_y2.copy()
        for res_unit in self.chainlist[::-1]:
            b_var = chainer.Variable(b)
            with chainer.force_backprop_mode():
                c_var = res_unit(b_var)
                c_var.grad = ga
                c_var.backward()
            a -= c_var.array
            gb += b_var.grad
            a, b = b, a
            ga, gb = gb, ga

        gx = xp.concatenate((ga, gb), axis=1)
        return gx,


class ResnetUnit(chainer.Chain):
    '''Residual unit (y = x + f(x)) of 'pre-activation'.
    This corresponds to the function F or G in the revnet paper.
    '''
    def __init__(self, channels):
        super(ResnetUnit, self).__init__(
            # In revnet training, BN's `decay` parameters should be `sqrt`ed
            # in order to compensate double forward passes for one update.
            brc1=BRCChain(channels, channels, 3, pad=1, decay=0.9**0.5),
            brc2=BRCChain(channels, channels, 3, pad=1, decay=0.9**0.5))

    def __call__(self, x):
        h = self.brc1(x)
        h = self.brc2(h)
        return h


def extend_channels(x, out_ch):
    '''Extends channels (i.e. depth) of the input BCHW tensor x by zero-padding
    if out_ch is larger than the number of channels of x, otherwise returns x.

    Note that this function is different from `functions.extend_channels` that
    pads a zero-filled tensor by concatenating it to the end of `x`
    as following:
        [1, 2, 3, 4] -> [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]
    On the other hand, this function is modified to fit to use with revnet that
    pads zeros as following:
        [1, 2, 3, 4] -> [1, 2, 0, 0, 0, 3, 4, 0, 0, 0]
    '''
    b, in_ch, h, w = x.shape
    if in_ch == out_ch:
        return x
    elif in_ch > out_ch:
        raise ValueError('out_ch must be larger than x.shape[1].')

    xp = chainer.cuda.get_array_module(x)
    x1, x2 = F.split_axis(x, 2, axis=1)
    filler_shape = (b, (out_ch - in_ch) // 2, h, w)
    filler = xp.zeros(filler_shape, x.dtype)
    return F.concat((x1, filler, x2, filler), axis=1)


if __name__ == '__main__':
    # Hyperparameters
    p = SimpleNamespace()
    p.gpu = 0  # GPU>=0, CPU < 0
    p.n = 18   # number of units in each stage
    p.num_epochs = 160
    p.batch_size = 100
    p.lr_init = 0.1
    p.lr_decrease_rate = 0.1
    p.weight_decay = 2e-4
    p.epochs_lr_divide10 = [80, 120]

    xp = np if p.gpu < 0 else chainer.cuda.cupy

    # Dataset
    train, test = load_cifar10_as_ndarray(3)
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)
    std_rgb = x_train.std((0, 2, 3), keepdims=True)
    x_train /= std_rgb
    x_test /= std_rgb
    mean_rgb = x_train.mean((0, 2, 3), keepdims=True)
    x_train -= mean_rgb
    x_test -= mean_rgb

    # Model and optimizer
    model = Revnet(n=p.n)
    if p.gpu >= 0:
        model.to_gpu()
#    optimizer = optimizers.MomentumSGD(p.lr_init)
    optimizer = optimizers.NesterovAG(p.lr_init)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(p.weight_decay))

    # Training loop
    train_loss_log = []
    train_acc_log = []
    test_loss_log = []
    test_acc_log = []
    best_test_acc = 0
    try:
        for epoch in range(p.num_epochs):
            if epoch in p.epochs_lr_divide10:
                optimizer.lr *= p.lr_decrease_rate

            epoch_losses = []
            epoch_accs = []
            for i in tqdm(range(0, num_train, p.batch_size)):
                x_batch = random_augment_padding(x_train[i:i+p.batch_size])
                x_batch = xp.asarray(x_batch)
                c_batch = xp.asarray(c_train[i:i+p.batch_size])
                model.cleargrads()
                with chainer.using_config('train', True):
                    y_batch = model(x_batch)
                    loss = F.softmax_cross_entropy(y_batch, c_batch)
                    acc = F.accuracy(y_batch, c_batch)
                    loss.backward()
                optimizer.update()
                epoch_losses.append(loss.data)
                epoch_accs.append(acc.data)

            epoch_loss = np.mean(chainer.cuda.to_cpu(xp.stack(epoch_losses)))
            epoch_acc = np.mean(chainer.cuda.to_cpu(xp.stack(epoch_accs)))
            train_loss_log.append(epoch_loss)
            train_acc_log.append(epoch_acc)

            # Evaluate the test set
            losses = []
            accs = []
            for i in tqdm(range(0, num_test, p.batch_size)):
                x_batch = xp.asarray(x_test[i:i+p.batch_size])
                c_batch = xp.asarray(c_test[i:i+p.batch_size])
                with chainer.using_config('train', False):
                    y_batch = model(x_batch)
                    loss = F.softmax_cross_entropy(y_batch, c_batch)
                    acc = F.accuracy(y_batch, c_batch)
                losses.append(loss.data)
                accs.append(acc.data)
            test_loss = np.mean(chainer.cuda.to_cpu(xp.stack(losses)))
            test_acc = np.mean(chainer.cuda.to_cpu(xp.stack(accs)))
            test_loss_log.append(test_loss)
            test_acc_log.append(test_acc)

            # Keep the best model so far
            if test_acc > best_test_acc:
                best_model = deepcopy(model)
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_epoch = epoch

            # Display the training log
            print('{}: loss = {}'.format(epoch, epoch_loss))
            print('test acc = {}'.format(test_acc))
            print('best test acc = {} (# {})'.format(best_test_acc,
                                                     best_epoch))
            print(p)

            plt.figure(figsize=(10, 4))
            plt.title('Loss')
            plt.plot(train_loss_log, label='train loss')
            plt.plot(test_loss_log, label='test loss')
            plt.legend()
            plt.grid()
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.title('Accucary')
            plt.plot(train_acc_log, label='train acc')
            plt.plot(test_acc_log, label='test acc')
            plt.legend()
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print('Interrupted by Ctrl+c!')

    print('best test acc = {} (# {})'.format(best_test_acc,
                                             best_epoch))
    print(p)
    print(optimizer)
