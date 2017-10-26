import copy
import unittest

import numpy

import chainer
import chainer.functions as F
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

import revnet


class TestRevnetStage(unittest.TestCase):

    def setUp(self):
        B = 10  # batch size
        C = 32
        H, W = 32, 32
        n = 20  # number of units in each stage

        self.chainlist = revnet.RevnetStage(n, C)

        self.x = numpy.random.uniform(
            -1, 1, (B, C, H, W)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (B, C, H, W)).astype(numpy.float32)
        self.check_backward_options = {
            'dtype': numpy.float32, 'atol': 1e-2, 'rtol': 5e-2, 'eps': 5e-4}

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            lambda _x: self.chainlist(_x),
            x_data, y_grad, **self.check_backward_options)

    @condition.retry(5)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(5)
    def test_backward_gpu(self):
        self.chainlist.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_allclose_backward_by_direct_and_reverse(self, x_data, y_grad):
        net_direct = copy.deepcopy(self.chainlist)
        x_direct = chainer.Variable(x_data)
        y_direct = forward(net_direct, x_direct)
        y_direct.grad = y_grad
        y_direct.backward()

        net_reverse = copy.deepcopy(self.chainlist)
        x_reverse = chainer.Variable(x_data)
        y_reverse = net_reverse(x_reverse)
        y_reverse.grad = y_grad
        y_reverse.backward()

        testing.assert_allclose(x_direct.grad, x_reverse.grad,
                                atol=1e-3, rtol=3e-0)

        for p_direct, p_reverse in zip(net_direct.params(),
                                       net_reverse.params()):
            testing.assert_allclose(p_direct.grad, p_reverse.grad,
                                    atol=1e-3, rtol=2e-0)

    @condition.retry(5)
    def test_check_allclose_backward_by_direct_and_reverse_cpu(self):
        self.check_allclose_backward_by_direct_and_reverse(self.x, self.gy)

    @attr.gpu
    @condition.retry(5)
    def test_check_allclose_backward_by_direct_and_reverse_gpu(self):
        self.chainlist.to_gpu()
        self.check_allclose_backward_by_direct_and_reverse(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


def forward(chainlist, x):
    x1, x2 = F.split_axis(x, 2, axis=1)

    for res_unit in chainlist:
        x2 += res_unit(x1)
        x1, x2 = x2, x1

    y = F.concat((x1, x2), axis=1)
    return y


testing.run_module(__name__, __file__)
