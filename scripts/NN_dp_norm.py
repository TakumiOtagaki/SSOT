from modules.nussinov.nussinov import nussinov, traceback
from modules.nussinov.onehot_differential_nussinov import logsumexp_nussinov, traceback_MAP,  onehot_Relu_basepair, onehot_Relu_basepair_grad, grad_dp_X
from modules.nussinov.onehot_differential_nussinov import onehot_basepair, onehot_basepair_grad
# from modules.ss.internal_distance import DijkstraSS
from modules.ss.util import bptuple2dotbracket
import numpy as np
import sys
import pandas as pd
import torch

from functools import partial
from scipy.special import softmax

from modules.noise import SumOfGammaNoiseDistribution

"""
二次構造の loss だと情報が減りすぎているから、欲しい二次構造 y の nussionv traceback route の一致具合をコストにする。
dijkstra のルートの話と似ている。
"""

# DEBUG = True
DEBUG = False


def normal_nussinov2dotbracket(rna_sequence):
    print("dp")
    dp = nussinov(rna_sequence)
    print("traceback")
    tuple_structure = traceback(
        dp, rna_sequence, 0, len(rna_sequence)-1, set())
    print("nussinov")
    print(tuple_structure)

    # dot bracket
    dot_bracket = ['.' for _ in range(len(rna_sequence))]
    for (i, j) in tuple_structure:
        i, j = min(i, j), max(i, j)
        dot_bracket[i] = '('
        dot_bracket[j] = ')'
    ss = ''.join(dot_bracket)
    return ss


def onehot2seq(X):
    seq = ""
    for i in range(X.size(0)):
        m = torch.argmax(X[i])
        if m == 0:
            seq += "A"
        elif m == 1:
            seq += "U"
        elif m == 2:
            seq += "G"
        elif m == 3:
            seq += "C"
    return seq


def dp2thetaphi(dp, n, x, T, threshold):
    theta = torch.zeros(n, n, 3)
    phi = torch.zeros(n, n, n)
    for i in range(n-1):
        for j in range(i+1, n):
            theta[i][j][0] = dp[i+1][j]
            theta[i][j][1] = dp[i][j-1]
            theta[i][j][2] = dp[i+1][j-1] + \
                onehot_Relu_basepair(x[i], x[j], T, threshold)
            for k in range(i+1, j):
                phi[i][j][k] = dp[i][k] + dp[k+1][j]
    return theta, phi


def random_vector(n, dim):  # 尖った分布。x[i] のうちある d でだけ値が大きくあって欲しい
    X = torch.zeros(n, dim)
    for i in range(n):
        X[i] = torch.rand(dim)
        X[i] = torch.exp(- (X[i] - 1) ** 2 / 0.5)

    return X


def train_model(n,  y_seq, x_init, lr, num_epochs, noise_scale, dim, T, ReluThreshold, gaussian_sigma):
    # bp = partial(onehot_Relu_basepair, T=T, threshold=ReluThreshold)
    # bp_grad = partial(onehot_Relu_basepair_grad, T=T, threshold=ReluThreshold)
    bp = partial(onehot_basepair, T=T, gaussian_sigma=gaussian_sigma)
    bp_grad = partial(onehot_basepair_grad, T=T, gaussian_sigma=gaussian_sigma)

    MAP = partial(traceback_MAP, n=n)

    lr = torch.tensor(lr, dtype=torch.float32)
    noise = SumOfGammaNoiseDistribution(k=3, nb_iterations=40)
    A = torch.zeros(n, n)
    count_no_grad = 0
    L_X_old = torch.zeros(n, dim)
    for i in range(n):
        for j in range(n):
            A[i, j] = torch.abs(torch.tensor(i - j))

    # initialization
    # n times 4 matrix (one hot)
    X = torch.zeros(n, dim)
    X.requires_grad = False
    # for i in range(n):
    #     # random initialization
    #     X[i] = torch.rand(dim) * 5
    #     # softmax
    #     X[i] = torch.exp(X[i]) / torch.sum(torch.exp(X[i]))

    # for test
    A, U, G, C = 0, 1, 2, 3
    dict_ = {"A": A, "U": U, "G": G, "C": C}
    # Xstr = "CCCAAACCCUUUAAA"
    Xstr = x_init
    for i in range(n):
        X[i][dict_[Xstr[i]]] = 1
    X0 = X.detach()
    seq0 = onehot2seq(X)
    print("X", pd.DataFrame(X.detach().numpy()))
    L_list = []

    for epoch in range(num_epochs):
        print(f"-------------------- Epoch: {epoch} ---------------------")

        # --------------------------- FORWARD ---------------------------
        dp = logsumexp_nussinov(X, T, gaussian_sigma, ReluThreshold)
        theta, phi = dp2thetaphi(dp, n, X, T, ReluThreshold)

        # MAP 推定 on traceback
        ss_tuple, z1, z2 = MAP(theta, phi)
        ss_simple_nussinov = normal_nussinov2dotbracket(onehot2seq(X))

        print(f"seq 1st = {seq0}")
        print(f"seq now = {onehot2seq(X)}")
        print(f"xss : {bptuple2dotbracket(n, ss_tuple)}")
        print(f"ss_by_discrete_nussinov:", ss_simple_nussinov)
        print(f"yss : {y_seq.ss_string}")
        print(f"y  : {y_seq.seq}")

        if ss_simple_nussinov == y_seq.ss_string:
            print("match!! found best x")

        L = torch.sum((y_seq.dp - dp) ** 2)
        L_prime = torch.sum((y_seq.discrete_dp * T - dp) ** 2)
        L_list.append(L.item())
        print("Loss L:", L)
        print("L prime:", L_prime)

        # ----- BACKWARD -----
        # Implement backpropagation by hand

        L_dp = 2 * (dp - y_seq.dp)

        #  discrete な値を使って微分
        # L_dp = 2 * (dp - y_seq.discrete_dp * T)
        # print("L_dp:", L_dp)
        L_X = torch.zeros(n, dim)

        # L_X
        ddpdX = grad_dp_X(dp, X, ReluThreshold, T, dim, gaussian_sigma)
        # ddpdX[n][dim][n][n]
        # for t in range(n):
        #     for d in range(dim):
        #         for i in range(n):
        #             for j in range(n):
        #                 L_X[t][d] += ddpdX[t][d][i][j] * L_dp[i][j]
        # アインシュタインの縮約記法
        L_X = torch.einsum("tdij,ij->td", ddpdX, L_dp)
        if DEBUG:
            # print("L_z1:", pd.DataFrame(L_z1[:, :, 2].detach().numpy()))
            print("theta:", pd.DataFrame(theta[:, :, 2].detach().numpy()))
            # print("z1 = ", pd.DataFrame(z1[:, :, 2].detach().numpy()))
            # print("new_z1 = ", pd.DataFrame(new_z1[:, :, 2].detach().numpy()))
            print("L_dp:", pd.DataFrame(L_dp.detach().numpy()))
            print("ddpdX:", ddpdX)
            print("L_X, X", pd.DataFrame(L_X.detach().numpy()), "\n",
                  pd.DataFrame(X.detach().numpy()))
        X_old = X.detach()
        L_X_old = L_X.detach()
        X = X - lr * L_X
        # X = softmax(X.detach().numpy(), axis=1)
        # Relu
        X = torch.nn.functional.relu(X)
        X = torch.nn.functional.normalize(X, p=1, dim=1)
        # X = softmax(X.detach().numpy(), axis=1)
        # X = torch.tensor(X, dtype=torch.float32)
        if epoch % 10 == 1:
            print("X:", pd.DataFrame(X.detach().numpy()))
        # sys.exit()

        # 内積
        # L_X とかを一次元ベクトルに
        print("L_old dot L_new, norm L_old, norm L_new", torch.dot(L_X_old.view(-1),
              L_X.view(-1)) / (torch.norm(L_X_old) * torch.norm(L_X)), torch.norm(L_X_old), torch.norm(L_X))

        # たまに確率的な勾配を足す
        if epoch % 25 == 0:
            L_prob = torch.zeros(n, dim)
            for i in range(n):
                for d in range(dim):
                    # random vector
                    L_prob[i][d] = torch.randn(1)
            X = X + L_prob / torch.sqrt(torch.tensor(epoch) + 1)
        # もし X が nan を含んでいたらエラーを吐く
        if torch.isnan(X).any():
            print("X has nan.")
            print("X:", pd.DataFrame(X.detach().numpy()))
            print("L_X:", pd.DataFrame(L_X.detach().numpy()))  # これも nan だった...
            sys.exit()
        # if X_old != X: tensor なので違う表記
        if onehot2seq(X) != onehot2seq(X_old):
            print("X has changed now at epoch", epoch)

    return X0, seq0, X,  onehot2seq(X), bptuple2dotbracket(n, ss_tuple), count_no_grad,  L_list


class Sequence:
    def __init__(self, seq):
        self.seq = seq
        self.len = len(seq)
        self.onehot = torch.zeros(self.len, 4)
        for i in range(self.len):
            if seq[i] == "A":
                self.onehot[i][0] = 1
            elif seq[i] == "U":
                self.onehot[i][1] = 1
            elif seq[i] == "G":
                self.onehot[i][2] = 1
            elif seq[i] == "C":
                self.onehot[i][3] = 1

    def nussinov_thetaphi(self, T, gaussian_sigma, ReluThreshold):
        dp = logsumexp_nussinov(self.onehot, T, gaussian_sigma, ReluThreshold)
        self.theta, self.phi = dp2thetaphi(
            dp, self.len, self.onehot, T, ReluThreshold)
        self.dp = dp
        # self.ss_string = bptuple2dotbracket(m, tuple_structure)
        self.calcss(self.theta, self.phi)

    def calcss(self, theta, phi):
        self.bptuple, self.z1, self.z2 = traceback_MAP(theta, phi, self.len)
        self.ss_string = bptuple2dotbracket(self.len, self.bptuple)

    def discrete_nussinov(self):
        discrete_dp = nussinov(self.seq)
        self.discrete_dp = torch.zeros(self.len, self.len)
        for i in range(self.len):
            for j in range(i+1, self.len):
                self.discrete_dp[i][j] = discrete_dp[i][j]


def main():
    # parameters about x
    dim = 4
    T = 4.0  # 小さすぎると logsumexp と max の近似の誤差が大きくなるので、 5 くらいが望ましい。大きすぎても nan を吐き出しかねないので注意
    noise_scale = 1e-2
    num_epochs = 350
    Relu_Threshold = 0.4  # should be larger than 0.25, smaller than 1.0
    lr = 1e-2
    gaussian_sigma = 0.2  # 不要なので None にしておく。今後使うかもなので一応残しておくけど。
    # y = "GAGCCAACAC"
    y = "GGGAACCCGGGAACCC"  # tRNA
    m = len(y)
    n = m
    x_init = "A" * n

    y_seq = Sequence(y)
    y_seq.nussinov_thetaphi(
        T, gaussian_sigma, Relu_Threshold)
    y_seq.discrete_nussinov()
    # print("y_seq.dp", y_seq.dp)
    # print("y_seq.ss_string:", y_seq.ss_string)

    X0, seq0, X, seq, x_ss, count_no_grad, L_list = train_model(n,  y_seq, x_init, lr, num_epochs,
                                                                noise_scale, dim, T, Relu_Threshold, gaussian_sigma)
    print(f"count_no_grad: {count_no_grad}")
    print(f"X0:\n", pd.DataFrame(X0.detach().numpy()))
    print("X: \n", pd.DataFrame(X.detach().numpy()))
    print(f"difference: {torch.sum((X - X0) ** 2)}")
    print(f"seq0: {seq0}")
    print(f"seq : {seq}")
    print(f"y   : {y_seq.seq}")
    print(f"xss : {x_ss}")
    print(f"yss : {y_seq.ss_string}")

    # epoch vs L, L'
    import matplotlib.pyplot as plt
    # x軸は epoch
    plt.plot(range(len(L_list) - 3), L_list[3:], label="L")
    plt.xlabel("epoch")
    plt.ylabel("L, L'")
    plt.title(f"L and L' vs epoch \n x0 = {x_init}, y = {y}")
    plt.legend()
    plt.savefig("/Users/ootagakitakumi/Library/Mobile Documents/com~apple~CloudDocs/大学院/浅井研究室/研究/InverseFolding/SSOT/figures/L_dploss.png")


if __name__ == "__main__":
    main()
