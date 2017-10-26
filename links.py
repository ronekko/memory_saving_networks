# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:47:11 2017

@author: sakurai
"""

import chainer
import chainer.functions as F
import chainer.links as L


class BRCChain(chainer.Chain):
    '''
    This is a composite link of sequence of BatchNormalization, ReLU and
    Convolution2D (a.k.a. pre-activation unit).
    '''
    def __init__(self, ch_in, ch_out, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, decay=0.9):
        super(BRCChain, self).__init__(
            bn=L.BatchNormalization(ch_in, decay=decay),
            conv=L.Convolution2D(ch_in, ch_out, ksize=ksize, stride=stride,
                                 pad=pad, nobias=nobias, initialW=initialW,
                                 initial_bias=initial_bias))

    def __call__(self, x):
        h = self.bn(x)
        h = F.relu(h)
        y = self.conv(h)
        return y
