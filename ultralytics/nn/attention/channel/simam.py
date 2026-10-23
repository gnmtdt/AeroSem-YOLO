"""
SimAM - Simple, Parameter-Free Attention Module

论文: SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
会议: ICML 2021
论文链接: https://proceedings.mlr.press/v139/yang21o/yang21o.pdf

基于能量函数的3D注意力机制，无需额外参数。通过计算每个神经元的能量函数推导注意力权重，
实现真正的零参数注意力。forward 返回 x * sigmoid(y)。
"""

import torch
import torch.nn as nn


class SimAM(torch.nn.Module):
    def __init__(self, dim, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


__all__ = ['SimAM']
