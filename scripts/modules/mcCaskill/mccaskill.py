import torch
import sys
import numpy as np
from functools import partial
# X は n times 4 matrix (torch)


def onehot_basepair(x_i, x_j, T, gaussian_sigma):
    # xi, xj はどちらも 0/1 の四次元ベクトル。A, U, G, C の順番に並んでいる。
    # if (xi, xj) is (A, U), (U, A), (G, C), (C, G), (U, G), (G, U): return 1
    # これを 内積で表現する（微分可能にするために）
    # 分散が sigma のガウス分布で平均が 1 の値を返す
    def gaussian(x, sigma):
        return torch.exp(- (x - 1) ** 2 / (sigma ** 2))
    s_i = gaussian(x_i, gaussian_sigma)
    s_j = gaussian(x_j, gaussian_sigma)
    A_U = s_i[0] * s_j[1] + s_i[1] * s_j[0]  # 1
    G_C = s_i[2] * s_j[3] + s_i[3] * s_j[2]  # 2
    U_G = s_i[1] * s_j[2] + s_i[2] * s_j[1]  # 3

    # pairs which do not form basepair
    A_G = s_i[0] * s_j[2] + s_i[2] * s_j[0]  # 4
    A_C = s_i[0] * s_j[3] + s_i[3] * s_j[0]  # 5
    U_C = s_i[1] * s_j[3] + s_i[3] * s_j[1]  # 6
    # return T * (A_U + G_C + U_G)
    return T * (A_U + G_C + U_G) * (1 - A_G) * (1 - A_C) * (1 - U_C)


def onehot_basepair_grad(x_i, x_j, T, gaussian_sigma):
    """
    derivative of onehot_basepair by x_i.
    """
    def g(x):
        return torch.exp(- (x - 1) ** 2 / (gaussian_sigma ** 2))
        # return torch.exp(- (x - 0.8) ** 2 / (gaussian_sigma ** 2))

    def dg(x):
        return - 2 * (x - 1) / (gaussian_sigma ** 2) * g(x)
        # return - 2 * (x - 0.8) / (sigma ** 2) * gaussian(x, sigma)
    grad = torch.zeros(4)
    s_j = g(x_j)
    s_i = g(x_i)
    A_U = s_i[0] * s_j[1] + s_i[1] * s_j[0]  # 1
    G_C = s_i[2] * s_j[3] + s_i[3] * s_j[2]  # 2
    U_G = s_i[1] * s_j[2] + s_i[2] * s_j[1]  # 3

    # pairs which do not form basepair
    A_G = s_i[0] * s_j[2] + s_i[2] * s_j[0]  # 4
    A_C = s_i[0] * s_j[3] + s_i[3] * s_j[0]  # 5
    U_C = s_i[1] * s_j[3] + s_i[3] * s_j[1]  # 6

    # grad[0] = dg(x_i[0], gaussian_sigma) * s_j[1]
    # grad[1] = dg(x_i[1], gaussian_sigma) * (s_j[0] + s_j[2])
    # grad[2] = dg(x_i[2], gaussian_sigma) * (s_j[3] + s_j[1])
    # grad[3] = dg(x_i[3], gaussian_sigma) * s_j[2]
    # return grad * T

    grad[0] = (1 - U_C) * (  # A で微分
        dg(x_i[0]) * s_j[1] * (1 - A_G) * (1 - A_C)
        - (A_U + U_G + G_C) * (dg(x_i[0]) * s_j[2] *
                               (1 - A_C) + dg(x_i[0]) * s_j[3] * (1 - A_G))
    )
    grad[1] = (1 - A_C) * (1 - A_G) * (  # U で微分
        (dg(x_i[1]) * (s_j[0] + s_j[2])) * (1 - U_C)
        - (A_U + U_G + G_C) * (dg(x_i[1] * s_j[3]))
    )
    grad[2] = (1 - A_C) * (1 - U_C) * (  # G で微分
        dg(x_i[2]) * (s_j[3] + s_j[1]) * (1 - A_G)
        - (A_U + U_G + G_C) * (dg(x_i[2]) * (s_j[0]))
    )
    grad[3] = (1 - A_G) * (  # C で微分
        dg(x_i[3]) * s_j[2] * (1 - A_C) * (1 - U_C)
        - (A_U + U_G + G_C) * (dg(x_i[3]) * \
                               (s_j[0] * (1 - U_C) + s_j[1] * (1 - A_C)))
    )

    return grad * T


def differential_mccaskill(X, T, gaussian_sigma):
    """
    X: n times 4 matrix
    T: temperature
    gaussian_sigma: variance of gaussian function
    """
    n = X.size(0) 
    