import math
from modules.markov.kmer_mm import KmerMarkovModel
# from modules.nussinov.stochastic_nussinov import StochasticNussinov
from modules.nussinov.onehot_differential_nussinov import logsumexp_nussinov, differential_traceback, simple_traceback
# from modules.ss.internal_distance import DijkstraSS
from modules.ss.internal_distance import DijkstraSS, floyd_warshall, dijkstra_route
from modules.ss.util import bptuple2dotbracket
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import ot
from modules.noise import SumOfGammaNoiseDistribution


DEBUG = False


def check_convergence(f, g, epsilon, C, p, q, threshold=1e-7):
    n, m = len(p), len(q)
    P = torch.exp((-C + g.reshape(1, m) + f.reshape(n, 1)) / epsilon)
    return torch.allclose(P.sum(0), q, atol=threshold) and torch.allclose(P.sum(1), p, atol=threshold)


def log_sinkhorn(p, q, C, epsilon, iter):
    epsilon = torch.tensor(epsilon, dtype=torch.float32, requires_grad=False)
    n, m = len(p), len(q)
    p = torch.tensor(ot.unif(n), dtype=torch.float32, requires_grad=False)
    q = torch.tensor(ot.unif(m), dtype=torch.float32, requires_grad=False)
    f = torch.zeros(n)
    g = torch.zeros(m)

    congergence = False
    for t in range(iter):
        f = -epsilon * \
            torch.logsumexp((-C + g.reshape(1, m)) / epsilon,
                            dim=1) + epsilon * torch.log(p)

        # for debugging: all shape
        # print(f"f.shape: {f.shape}, C.shape: {C.shape}, torch.logsumexp((-C + f.reshape(n, 1)) / epsilon,dim=0).shape: {torch.logsumexp((-C + f.reshape(n, 1)).T / epsilon,dim=0).shape}, epsilon * torch.log(q).shape: {(epsilon * torch.log(q)).shape}")

        g = -epsilon * \
            torch.logsumexp((-C + f.reshape(n, 1)) / epsilon,
                            dim=0) + epsilon * torch.log(q)

        if check_convergence(f, g, epsilon, C, p, q):
            congergence = True
            break

    P = torch.exp((-C + g.reshape(1, m) + f.reshape(n, 1)) / epsilon)
    if not congergence:
        print("Not converged")
    return P


def compute_gromov_wasserstein_term(a, b, Cx, Cy, P):
    """
    Compute the Gromov-Wasserstein discrepancy term.

    Args:
    - ax, bx (numpy.ndarray): Weights for the source space X
    - dX (numpy.ndarray): Distance matrix in the source space X
    - ay, by (numpy.ndarray): Weights for the target space Y
    - dY (numpy.ndarray): Distance matrix in the target space Y
    - P (numpy.ndarray): Optimal transport plan between X and Y

    Returns:
    - float: Computed discrepancy term
    """
    # Calculate the weighted sum of squared distances for X
    weighted_dX_squared = torch.sum(a[:, None] * a[None, :] * (Cx ** 2))

    # Calculate the weighted sum of squared distances for Y
    weighted_dY_squared = torch.sum(b[:, None] * b[None, :] * (Cy ** 2))

    cross_term = torch.einsum('ij,ik,jl,kl->', P, Cx, Cy, P)

    # Combine terms
    result = weighted_dX_squared + weighted_dY_squared - 2 * cross_term
    return result


def compute_gw_transport(Cx, Cy, p, q, epsilon, lambda_, max_iter, sinkhorn_iter):
    # initialize randomly
    n, m = len(p), len(q)
    P_t = torch.ones(n, m) / (n * m)

    for t in range(max_iter):
        # C_t = -4 * np.dot(np.dot(Cx, P_t), Cy) # tensor に対応するように書き換える
        C_t = -4 * torch.mm(torch.mm(Cx, P_t), Cy) + (epsilon - lambda_) * P_t
        P_t = log_sinkhorn(p, q, C_t, epsilon=lambda_, iter=sinkhorn_iter)
    return P_t


def calculateDistmat(rna_ss_matrix, alpha, A):  # excluding pseudoknot!
    # rna secondary structure について、各塩基間の距離を計算する
    n = len(rna_ss_matrix)
    return (torch.ones(n, n) - rna_ss_matrix) * A + alpha * rna_ss_matrix


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
    print(seq)
    return seq


def train_model(n, m, alpha, Cy, ss_y, lr=1e-3, num_epochs=100, noise_scale=0.1, dim=4, T=1.0):
    # --- pseudo code ---
    noise = SumOfGammaNoiseDistribution(k=3, nb_iterations=40)
    A = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            A[i, j] = torch.abs(torch.tensor(i - j))

    # initialization
    # n times 4 matrix (one hot)
    X = torch.zeros(n, dim)
    X.requires_grad = True
    mask = torch.zeros(n, dim)
    # random に 1 を入れる

    # random0123seq
    random_seq = np.random.randint(0, dim, n)
    for i in range(n):
        mask[i, random_seq[i]] = 1

    X = X + mask
    print(f"seq = {onehot2seq(X)}")

    for epoch in range(num_epochs):
        # 全ての grad の初期化
        X.grad = None

        print(
            f"----------------------------- Epoch: {epoch} --------------------------")
        # noise generation
        noise_W = noise.sample((n, n)) * noise_scale
        noise_dp = noise.sample((n, n)) * noise_scale

        # ----- FORWARD -----
        # Stochastic Nussinov
        dp = logsumexp_nussinov(X, T)
        if DEBUG:
            print("dp = ", dp)

        # MAP 推定 on traceback
        ss_tuple = simple_traceback(dp + noise_dp, X, T)
        rna_ss_matrix = torch.zeros(n, n)
        for i, j in ss_tuple:
            rna_ss_matrix[i, j] = rna_ss_matrix[j, i] + 1
            rna_ss_matrix[j, i] = rna_ss_matrix[j, i] + 1

        ss_dot_bracket = bptuple2dotbracket(n, ss_tuple)
        print(f"ss : {ss_dot_bracket}")
        print(f"y  : {ss_y}")

        # convert rna_ss_matrix to ss matrix
        W = calculateDistmat(rna_ss_matrix, alpha, A)

        Cx = floyd_warshall(W + noise_W)

        # Cx を tensor にして、Loss (= GW) の勾配を計算できるようにする
        Cx = torch.tensor(Cx, requires_grad=True, dtype=torch.float32)

        # GW distance
        p = torch.tensor(ot.unif(n))
        q = torch.tensor(ot.unif(m))
        P = compute_gw_transport(Cx, Cy, p, q, epsilon=1e-2, lambda_=10,
                                 max_iter=50, sinkhorn_iter=1000)
        GW = compute_gromov_wasserstein_term(p, q, Cx, Cy, P)

        print(f"---------- GW distance: {GW} ----------")

        # ----- BACKWARD -----
        # Implement backpropagation by hand

        GW.backward()

        # delta Cx
        L_Cx = Cx.grad.detach().numpy()

        # delta W
        W_new = W - lr * L_Cx
        L_W = torch.zeros(n, n)

        print("dijkstra n^2 times")
        for i in range(n):
            for j in range(i+1, n):
                R_newij = dijkstra_route(W_new + noise_W, i, j)
                R_oldij = dijkstra_route(W + noise_W, i, j)
                L_W += 2 * (R_newij - R_oldij)

        # nussinov traceback の勾配を計算する
        # L_z1 を考える
        L_rna_ss_matrix = (alpha - A) * L_W

        dp_new = dp - lr * L_rna_ss_matrix
        new_rna_ss_matrix = torch.zeros(n, n)
        for i, j in simple_traceback(dp_new, X, T):
            new_rna_ss_matrix[i, j] = 1
            new_rna_ss_matrix[j, i] = 1
        L_dp = rna_ss_matrix - new_rna_ss_matrix

        # L_X
        # dp_ij に対して X[i] の勾配を計算する
        L_X = torch.zeros(n, dim)
        for i in range(n):
            for j in range(i+1, n):
                dp_grad_ij = torch.autograd.grad(
                    dp[i][j], X, retain_graph=True)[0]
                L_X += dp_grad_ij * L_dp[i][j]

        # print("dp_grad:", dp_grad)
        print("grad_X = ", L_X)
        # X.backward()
        X = X - lr * L_X


def test_model(model):
    pass


def main():
    alpha = 0.5

    # pre calculation for loss function
    y = "((.....((...))....)...)"
    ss_y = [(0, 22), (1, 18), (7, 12), (8, 11)]
    ss_matrix_y = np.zeros((len(y), len(y)), dtype=int)
    for i, j in ss_y:
        ss_matrix_y[i][j] = 1
        ss_matrix_y[j][i] = 1
    y_dj = DijkstraSS(alpha=alpha)
    y_dj.calculateDistmat(ss_matrix_y)
    m = len(y)
    Cy = y_dj.dijkstra()
    # cy should be torch
    Cy = torch.tensor(Cy, dtype=torch.float32)

    # parameters about x
    n = 20
    T = 3.0
    lr = torch.tensor(1e-3, requires_grad=False, dtype=torch.float32)

    X = train_model(n, m,  alpha, Cy, y, lr, num_epochs=100,
                    noise_scale=0.1, dim=4, T=T)


if __name__ == "__main__":
    main()
