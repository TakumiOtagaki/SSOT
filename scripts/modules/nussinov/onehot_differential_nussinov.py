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

    # A_U = x_i[0] * x_j[1] + x_i[1] * x_j[0]  # 1
    # # print("A_U:", A_U)
    # G_C = x_i[2] * x_j[3] + x_i[3] * x_j[2]  # 2
    # # print("G_C:", G_C)
    # U_G = x_i[1] * x_j[2] + x_i[2] * x_j[1]  # 3
    # print("U_G:", U_G)
    return (A_U + G_C + U_G) * T


def onehot_basepair_grad(x_i, x_j, T, gaussian_sigma):
    """
    derivative of onehot_basepair by x_i.
    """
    def gaussian(x, sigma):
        return torch.exp(- (x - 1) ** 2 / (sigma ** 2))

    def derivative_gaussian(x, sigma):
        return - 2 * (x - 1) / (sigma ** 2) * gaussian(x, sigma)
    grad = torch.zeros(4)
    s_j = gaussian(x_j, gaussian_sigma)

    grad[0] = derivative_gaussian(x_i[0], gaussian_sigma) * s_j[1]
    grad[1] = derivative_gaussian(x_i[1], gaussian_sigma) * (s_j[0] + s_j[2])
    grad[2] = derivative_gaussian(x_i[2], gaussian_sigma) * (s_j[3] + s_j[1])
    grad[3] = derivative_gaussian(x_i[3], gaussian_sigma) * s_j[2]

    return grad * T


def onehot_Relu_basepair(x_i, x_j, T, threshold):
    # 勾配消失に対応
    A_U = torch.relu(x_i[0] * x_j[1] + x_i[1] * x_j[0] - threshold)  # 1
    G_C = torch.relu(x_i[2] * x_j[3] + x_i[3] * x_j[2] - threshold)  # 2
    U_G = torch.relu(x_i[1] * x_j[2] + x_i[2] * x_j[1] - threshold)  # 3
    return (A_U + G_C + U_G) * T


def onehot_Relu_basepair_grad(x_i, x_j, T, threshold):
    # 勾配消失に対応
    grad = torch.zeros(4)
    grad[0] += x_j[1] * (x_i[0] * x_j[1] + x_i[1] * x_j[0] > threshold)  # AU
    grad[1] += x_j[0] * (x_i[0] * x_j[1] + x_i[1] * x_j[0] > threshold)  # UA
    grad[2] += x_j[3] * (x_i[2] * x_j[3] + x_i[3] * x_j[2] > threshold)  # GC
    grad[3] += x_j[2] * (x_i[2] * x_j[3] + x_i[3] * x_j[2] > threshold)  # CG
    grad[1] += x_j[2] * (x_i[1] * x_j[2] + x_i[2] * x_j[1] > threshold)  # UG
    grad[2] += x_j[1] * (x_i[1] * x_j[2] + x_i[2] * x_j[1] > threshold)  # GU
    return grad * T


def res2onehot(res):
    if res == "A":
        return torch.tensor([1, 0, 0, 0])
    elif res == "U":
        return torch.tensor([0, 1, 0, 0])
    elif res == "G":
        return torch.tensor([0, 0, 1, 0])
    elif res == "C":
        return torch.tensor([0, 0, 0, 1])
    else:
        raise ValueError("Invalid input")


def logsumexp_nussinov(X, T, gaussian_sigma, RELU_THRESHOLD):
    # Fill the DP table
    n = X.size(0)
    dp = torch.full((n, n), 0.0, requires_grad=True)
    grad = torch.zeros(n, 4, requires_grad=True)
    for length in range(4, n + 1):  # Minimum loop length condition
        for i in range(n - length + 1):
            j = i + length - 1
            # if can_pair(i, j, rna_seq):
            #     dp[i][j] = max(dp[i][j], dp[i+1][j-1] + 1)
            # dp[i][j] = torch.logsumexp(
            # [
            #         dp[i+1][j],
            #         dp[i][j-1],
            #         dp[i+1][j-1] + onehot_basepair(X[i], X[j])]
            #     + [dp[i][k] + dp[k+1][j] for k in range(i+1, j)]
            # )
            dp_ij = torch.logsumexp(
                torch.stack([
                    dp[i+1][j],
                    dp[i][j-1],
                    # dp[i+1][j-1] +
                    # onehot_basepair(
                    #     X[i], X[j], T, gaussian_sigma) * (j - i >= 3)
                    dp[i+1][j-1] + onehot_Relu_basepair(
                        X[i], X[j], T, RELU_THRESHOLD) * (j - i >= 3)
                ]
                    + [dp[i][k] + dp[k+1][j] for k in range(i+1, j)]), dim=0
            )

            mask = torch.zeros(n, n)
            mask[i, j] = 1
            dp = dp + mask * dp_ij

    return dp


def calc_intermediate_params(dp, X, T, gaussian_sigma):
    n = dp.size(0)
    theta = torch.zeros(n, n, 3)  # i, j to i', j'
    phi = torch.zeros(n, n, n)  # i, j to i, k and k+1, j

    # i, j, i+1, j と i,j,i,j-1 と i,j,i+1,j-1 だけ theta の値を dp[*,*] に変更
    for i, j in [(i, j) for i in range(j) for j in range(n)]:
        # index error に注意して書く↓
        theta[i, j, 0] = dp[i+1][j]
        theta[i, j, 1] = dp[i][j-1]
        theta[i, j, 2] = dp[i+1][j-1] + \
            onehot_basepair(X[i], X[j], T, gaussian_sigma)
        for k in range(i+1, j):
            phi[i, j, k] = dp[i][k] + dp[k+1][j]
    return theta, phi


def differential_traceback(dp, X, T):
    # MAP 推定する
    # 基本的には traceback と同じで良い
    n = X.size(0)
    structure = set()  # tuple of (i, j)

    # theta = torch.zeros(n, n, 3)  # i, j to i', j'
    # phi = torch.zeros(n, n, n)  # i, j to i, k and k+1, j

    z1 = torch.zeros(n, n, 3)
    print("memory allocated")
    z1.requires_grad = True
    z2 = None
    # z2.requires_grad = True

    def traceback_(i, j):
        nonlocal structure
        nonlocal z1
        # nonlocal z2

        if i >= j:
            return
        choices = torch.stack([
            dp[i+1][j],
            dp[i][j-1],
            dp[i+1][j-1] + onehot_basepair(X[i], X[j], T) * (j - i > 3),
        ] + [dp[i][k] + dp[k+1][j] for k in range(i+1, j)]).unsqueeze(0)

        # print("CHOICE:")
        # print(choices)

        # 最大値のインデックスを取得
        max_index = torch.argmax(choices).item()
        # sys.exit()
        # print(max_index)
        A = torch.zeros(n, n, 3)

        if max_index == 0:  # dp[i+1][j]が最大
            A = torch.zeros(n, n, 3)
            A[i, j, 0] = 1
            z1 = z1 + A
            traceback_(i+1, j)
        elif max_index == 1:  # dp[i][j-1]が最大
            A = torch.zeros(n, n, 3)
            A[i, j, 1] = 1
            z1 = z1 + A
            traceback_(i, j-1)
        elif max_index == 2:  # dp[i+1][j-1] + onehot_basepairが最大
            A = torch.zeros(n, n, 3)
            A[i, j, 2] = 1
            z1 = z1 + A
            structure.add((i, j))
            traceback_(i+1, j-1)
        else:  # dp[i][k] + dp[k+1][j]の中で最大
            k = max_index - 3  # 調整したインデックス
            if not (i <= k <= j):
                print("i, k, j:", i, k, j)
                raise ValueError("k is out of range")
            # z2 = z2 + B
            traceback_(i, k)
            traceback_(k+1, j)

    traceback_(0, n-1)
    return structure, z1, z2


def simple_traceback(dp, X, T, gaussian_sigma):
    n = X.size(0)
    structure = set()  # tuple of (i, j)

    def traceback_(i, j):
        nonlocal structure
        if i >= j:
            return
        choices = torch.stack([
            dp[i+1][j],
            dp[i][j-1],
            (dp[i+1][j-1] +
                onehot_basepair(X[i], X[j], T, gaussian_sigma)) * (j - i >= 3),
        ] + [dp[i][k] + dp[k+1][j] for k in range(i+1, j)]).unsqueeze(0)
        # print("choices:", choices)
        max_index = torch.argmax(choices).item()
        if max_index == 2:
            print("bp: i, j:", i, j)
            structure.add((i, j))
            traceback_(i+1, j-1)
        elif max_index == 0:
            traceback_(i+1, j)
        elif max_index == 1:
            traceback_(i, j-1)
        elif max_index == 3 + (j - (i+1)):  # end
            print("全て 1e-20 以下でおかしい")
        else:
            k = max_index - 3 + i
            traceback_(i, k)
            traceback_(k+1, j)

    traceback_(0, n-1)
    return structure


def traceback_MAP(theta, phi, n):
    # theta: n, n, 3
    # phi: n, n, n
    z1 = torch.zeros(n, n, 3)
    z2 = torch.zeros(n, n, n)

    structure = set()  # tuple of (i, j)

    def traceback_(i, j):
        nonlocal z1
        nonlocal z2
        nonlocal structure
        if i >= j:
            return
        choices = torch.stack([theta[i, j, k] for k in range(3)] +
                              [phi[i, j, k] for k in range(i+1, j)])
        if j - i < 3:
            choices[2] = -np.inf
        max_index = torch.argmax(choices).item()
        if max_index == 0:
            z1[i, j, 0] += 1
            traceback_(i+1, j)
        elif max_index == 1:
            z1[i, j, 1] += 1
            traceback_(i, j-1)
        elif max_index == 2:
            z1[i, j, 2] += 1
            structure.add((i, j))
            traceback_(i+1, j-1)
        else:
            k = max_index - 3 + i + 1
            z2[i, j, k] += 1
            traceback_(i, k)
            traceback_(k+1, j)

    traceback_(0, n-1)
    return structure, z1, z2


def grad_dp_X(dp, X, ReluThreshold, T, dim, gaussian_sigma):
    # partial
    # bp = partial(onehot_Relu_basepair, T=T, threshold=ReluThreshold)
    # bp_grad = partial(onehot_Relu_basepair_grad, T=T, threshold=ReluThreshold)
    bp = partial(onehot_basepair, T=T, gaussian_sigma=gaussian_sigma)
    bp_grad = partial(onehot_basepair_grad, T=T, gaussian_sigma=gaussian_sigma)
    n = X.size(0)

    A = torch.zeros(n, dim, n, n)
    E = torch.exp(dp)  # E は dp の exp

    for length in range(4, n + 1):
        for i in range(n - length + 1):
            # i,j = (0, 3), (0, 4), (0, 5), (0, 6), (1, 4), (1, 5), (1, 6), (2, 5), (2, 6), (3, 6)
            j = i + length - 1  # (i, j) = (i, i+length-1), ..., (i, n-1)

            G_ij = torch.stack(
                [E[i+1][j], E[i][j-1], torch.exp(dp[i+1][j-1] + bp(X[i], X[j]))])
            G_ij = torch.cat(
                (G_ij, torch.stack([torch.exp(dp[i][k] + dp[k+1][j]) for k in range(i+1, j)])))
            G_ij_sum = G_ij.sum()

            for d in range(dim):
                for t in range(n):
                    numerator = torch.tensor(0.0)
                    # dp[i+1][j-1] + bp(X[i], X[j]) の微分
                    if i <= t <= j:
                        if t == i:
                            numerator += G_ij[2] * bp_grad(X[i], X[j])[d]
                        elif t == j:
                            numerator += G_ij[2] * bp_grad(X[j], X[i])[d]
                        else:
                            numerator += G_ij[2] * A[t, d, i+1, j-1]
                    else:
                        "微分は 0 になる"
                    # dp[i+1][j] の微分
                    if i+1 <= t <= j:
                        numerator += G_ij[0] * A[t, d, i+1, j]
                    # dp[i][j-1] の微分
                    if i <= t <= j-1:
                        numerator += G_ij[1] * A[t, d, i, j-1]
                    # dp[i][k] + dp[k+1][j] の微分
                    tmp_k = 0
                    for k in range(i+1, j):
                        if i <= t <= k:
                            tmp_k += A[t, d, i, k]
                        if k+1 <= t <= j:
                            tmp_k += A[t, d, k+1, j]
                        numerator += torch.exp(dp[i][k] + dp[k+1][j]) * tmp_k

                    A[t, d, i, j] += numerator / G_ij_sum

    return A


if __name__ == "__main__":
    x1, x2 = "A", "A"
    s1 = res2onehot(x1)
    s2 = res2onehot(x2)
    print(onehot_basepair(s1, s2))
