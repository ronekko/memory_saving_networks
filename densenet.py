# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:49:12 2017

@author: sakurai

A memory-saving implementation of DenseNet-BC for Chainer.
"Densely Connected Convolutional Networks"
https://arxiv.org/abs/1608.06993v3

Note that in this implementation, each unit in a dense block outputs a
concatenated tensor rather than the unit has concat-function at the beginning.

This implementation does not use the "shared memory technique" prsented in
"Memory-Efficient Implementation of DenseNets", because it uses "concat-first
units" that are slow due to tha backward of concat with many input variables.
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


class Densenet(chainer.Chain):
    '''
    Args:
        nums_units (list of int):
            List of numbers of primitive functions, for each of dense-blocks.

        growth_rate (int):
            Output channels of each primitive H(x), i.e. `k`.
    '''
    def __init__(self, num_classes=10, nums_units=[20, 20, 20], growth_rate=12,
                 dropout_rate=0.2, compression_factor=0.5):
        # TODO: Support dropout.
        if dropout_rate != 0:
            raise NotImplementedError(
                'Dropout is not supported yet. Set `dropout_rate` to 0.')

        super(Densenet, self).__init__()

        ch_in_init = 3
        ch_out_init = 2 * growth_rate

        ch_in_block = []
        ch_out_block = []
        for block, n in enumerate(nums_units):
            if block == 0:
                ch_in_block.append(ch_out_init)
            else:
                ch_in_block.append(int(ch_out_block[-1] * compression_factor))
            ch_out_block.append(ch_in_block[-1] + n * growth_rate)

        with self.init_scope():
            self.conv_init = L.Convolution2D(
                ch_in_init, ch_out_init, 3, pad=1, nobias=True)

            self.block_0 = DenseBlock(
                ch_in_block[0], nums_units[0], growth_rate, dropout_rate)
            self.trans_0 = TransitionLayer(ch_out_block[0], ch_in_block[1])

            self.block_1 = DenseBlock(
                ch_in_block[1], nums_units[1], growth_rate, dropout_rate)
            self.trans_1 = TransitionLayer(ch_out_block[1], ch_in_block[2])

            self.block_2 = DenseBlock(
                ch_in_block[2], nums_units[2], growth_rate, dropout_rate)
            self.trans_2 = TransitionLayer(ch_out_block[2], num_classes,
                                           global_pool=True)
        self._num_classes = num_classes

    def __call__(self, x):
        h = self.conv_init(x)
        h = self.block_0(h)
        h = self.trans_0(h)
        h = self.block_1(h)
        h = self.trans_1(h)
        h = self.block_2(h)
        h = self.trans_2(h)
        return h.reshape((-1, self._num_classes))


class DenseBlock(chainer.ChainList):
    def __init__(self, in_channels, num_units, growth_rate=12, drop_rate=0.2):
        '''
        Args:
            in_channels (int):
                Input channels of the block.
            num_units (int):
                Number of primitive functions, i.e. H(x), in the block.
            grouth_rate (int):
                Hyper parameter `k` which is output channels of each H(x).
            drop_rate (int):
                Drop rate for dropout.
        '''

        units = []
        for i in range(num_units):
            units += [DenseUnitBottleneck(in_channels, growth_rate)]
            in_channels = in_channels + growth_rate
        super(DenseBlock, self).__init__(*units)
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate

    def __call__(self, x):
        dense_block_function = DenseBlockFunction(
            self, self.growth_rate, self.drop_rate)
        y = dense_block_function(x)
        return y


class DenseBlockFunction(chainer.Function):
    def __init__(self, chainlist, growth_rate, drop_rate=0.0):
        """
        Args:
            chainlist (chainer.Chainlist):
                A ChainList of dense block.
        """
        self.chainlist = chainlist
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate

    def forward(self, inputs):
        x = chainer.Variable(inputs[0])
        with chainer.no_backprop_mode():
            for dense_unit in self.chainlist:
                h = dense_unit(x)
#                h = F.dropout(dense_unit(x), self.drop_rate)
                x = F.concat((x, h), axis=1)

        self.retain_outputs((0,))
        return x.array,

    def backward(self, inputs, grads):
        y_array = self.output_data[0]
        grad_y = grads[0]

        y1_array = y_array.copy()
        grad_y1 = grad_y.copy()
        for dense_unit in self.chainlist[::-1]:
            # TODO: compare speed and memory between two method below
            # (e.g. which of split or slice is faster or lighter)
#            # implemented as split
#            y1_array, y2_array = xp.split(y1_array, [y1_array.shape[1]-self.growth_rate], axis=1)
#            grad_y1, grad_y2 = xp.split(grad_y1, [grad_y1.shape[1]-self.growth_rate], axis=1)
            # implemented as slicing
            y1_array = y1_array[:, :-self.growth_rate]
            # Don't swap two lines below. Taking gy2 first then gy1 is correct.
            grad_y2 = grad_y1[:, -self.growth_rate:]
            grad_y1 = grad_y1[:, :-self.growth_rate]

            y1_var = chainer.Variable(y1_array)
            with chainer.force_backprop_mode():
                c_var = dense_unit(y1_var)
                # TODO: To conside dropout, reconstruct the mask from y2_array
                # here and multiply it to grad_y2. Note that tha mask consists
                # of 0 and 1/(1-drop_rate), not 0 and 1.
                c_var.grad = grad_y2
                c_var.backward()
            grad_y1 += y1_var.grad

        return grad_y1,


class DenseUnitBottleneck(chainer.Chain):
    def __init__(self, in_channels, out_channels, **kwargs):
        bottleneck = 4 * out_channels
        super(DenseUnitBottleneck, self).__init__(
            # `decay` parameter of BNs in DenseBlock should be `sqrt`ed
            # in order to compensate double forward passes for one update.
            brc1=BRCChain(in_channels, bottleneck,
                          ksize=1, pad=0, nobias=True, decay=0.9**0.5),
            brc3=BRCChain(bottleneck, out_channels,
                          ksize=3, pad=1, nobias=True, decay=0.9**0.5))

    def __call__(self, x):
        h = self.brc1(x)
        y = self.brc3(h)
        return y


class TransitionLayer(chainer.Chain):
    def __init__(self, in_channels, out_channels, global_pool=False):
        super(TransitionLayer, self).__init__(
            brc=BRCChain(in_channels, out_channels, ksize=1))
        self.global_pool = global_pool

    def __call__(self, x):
        h = self.brc(x)
        if self.global_pool:
            ksize = h.shape[2:]
        else:
            ksize = 2
        y = F.average_pooling_2d(h, ksize)
        return y


if __name__ == '__main__':
    # Hyperparameters
    p = SimpleNamespace()
    p.gpu = 0
    p.num_classes = 10
    p.nums_units = [16, 16, 16]
    p.growth_rate = 24  # out channels of each primitive funcion in dense block
    p.dropout_rate = 0.0
    p.num_epochs = 300
    p.batch_size = 100
    p.lr_init = 0.1
    p.lr_decrease_rate = 0.1
    p.weight_decay = 1e-4
    p.max_expand_pixel = 8
    p.epochs_lr_divide10 = [150, 225]

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
    model = Densenet(p.num_classes, nums_units=p.nums_units,
                     growth_rate=p.growth_rate, dropout_rate=p.dropout_rate)
    if p.gpu >= 0:
        model.to_gpu()
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
            perm = np.random.permutation(num_train)
            index_batches = np.split(perm, num_train // p.batch_size)
            for i_batch in tqdm(index_batches):
                x_batch = random_augment_padding(x_train[i_batch])
                x_batch = xp.asarray(x_batch)
                c_batch = xp.asarray(c_train[i_batch])
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
                with chainer.no_backprop_mode():
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
