import math
from modules.markov.kmer_mm import KmerMarkovModel
from modules.nussinov.stochastic_nussinov import StochasticNussinov
from modules.ss.internal_distance import DijkstraSS

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import ot
from modules.noise import SumOfGammaNoiseDistribution


def check_convergence(f, g, epsilon, C, p, q, threshold=1e-6):
    n, m = len(p), len(q)
    P = torch.exp((-C + g.reshape(1, m) + f.reshape(n, 1)) / epsilon)
    return torch.allclose(P.sum(0), q, atol=threshold) and torch.allclose(P.sum(1), p, atol=threshold)


def log_sinkhorn(p, q, C, epsilon=0.2, lambda_=1e-1, iter=100):
    epsilon = torch.tensor(epsilon, dtype=torch.float32, requires_grad=False)
    n, m = len(p), len(q)
    p = torch.tensor(p, dtype=torch.float32, requires_grad=False)
    q = torch.tensor(q, dtype=torch.float32, requires_grad=False)
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
            print("Converged at iteration: ", t)
            break

    P = torch.exp((-C + g.reshape(1, m) + f.reshape(n, 1)) / epsilon)
    if not congergence:
        print("Not converged")
    return P


def sinkhorn(p, q, C, epsilon=0.2, lambda_=1e-1, iter=100):
    # initialize randomly
    n, m = len(p), len(q)

    p = torch.tensor(p, dtype=torch.float32, requires_grad=False)
    q = torch.tensor(q, dtype=torch.float32, requires_grad=False)

    n, m = len(p), len(q)
    K = torch.exp(-C / epsilon)

    u = torch.ones(n) / n

    for t in range(iter):
        print(K.T @ u)
        # 要素ごとにわる
        v = q / (K.T @ u)
        u = p / (K @ v)

    P_t = u.reshape(n, 1) * K * v.reshape(1, m)

    return P_t


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
    print(weighted_dX_squared)

    # Calculate the weighted sum of squared distances for Y
    weighted_dY_squared = torch.sum(b[:, None] * b[None, :] * (Cy ** 2))

    print(weighted_dY_squared)

    cross_term = torch.einsum('ij,ik,jl,kl->', P, Cx, Cy, P)

    # Combine terms
    result = weighted_dX_squared + weighted_dY_squared - 2 * cross_term
    return result


def compute_gw_transport(Cx, Cy, p, q, epsilon=0.1, lambda_=1e-3, max_iter=50, sinkhorn_iter=100):
    # initialize randomly
    n, m = len(p), len(q)
    P_t = torch.ones(n, m) / (n * m)

    for t in range(max_iter):
        # print(f"t: {t}")
        # compute the gradient
        # C_t = -4 * np.dot(np.dot(Cx, P_t), Cy) # tensor に対応するように書き換える
        C_t = -4 * torch.mm(torch.mm(Cx, P_t), Cy) + (epsilon - lambda_) * P_t

        # P_t = ot.sinkhorn(hist_blue, hist_green, cost_matrix, reg=0.001, numItermax=100, method='sinkhorn_log')
        # P_t = sinkhorn(p, q, C_t, epsilon=epsilon,
        #                lambda_=lambda_, iter=sinkhorn_iter)
        P_t = log_sinkhorn(p, q, C_t, epsilon=epsilon,
                           lambda_=lambda_, iter=sinkhorn_iter)
    return P_t


def calculateDistmat(rna_ss_matrix, alpha):  # excluding pseudoknot!
    # rna secondary structure について、各塩基間の距離を計算する
    n = len(rna_ss_matrix)
    # インデックスの差の絶対値を計算するための行列を作成
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])

    # 条件に基づく値の割り当て
    A = np.where(rna_ss_matrix == 0, dist, alpha)

    return A


def train_model(kmer_size, n, m, alpha, Cy, lr=1e-3, num_epochs=100, noise_scale=0.1):
    # --- pseudo code ---
    noise = SumOfGammaNoiseDistribution(k=3, nb_iterations=40)
    # initialization
    num_states = 4 ** kmer_size
    pi = np.ones(num_states)
    transition_scores = np.ones((num_states, 4))

    for epoch in range(num_epochs):
        # ----- FORWARD -----

        # Kmer Markov Model
        noise_pi = noise.sample(pi.shape).numpy() * noise_scale
        noise_transition_scores = noise.sample(
            transition_scores.shape).numpy() * noise_scale

        pi = pi + noise_pi
        transition_scores = transition_scores + noise_transition_scores

        kmm = KmerMarkovModel(kmer_size, pi, transition_scores)
        _, __, X = kmm.MAP_sequence(n)

        # Stochastic Nussinov
        noise_X = noise.sample(X.shape).numpy() * noise_scale
        X += noise_X
        sn = StochasticNussinov(X)
        sn.nussinov()
        ss_tuple = sn.structure
        ss = sn.structure_matrix()

        # ss matrix to weighted matrix
        W = calculateDistmat(ss, alpha)

        # add noeise to W
        noise_W = noise.sample(W.shape).numpy() * noise_scale
        W = W + noise_W
        dijkstra = DijkstraSS(alpha=alpha, distmat=W)
        Cx = dijkstra.dijkstra()

        # Cx を tensor にして、Loss (= GW) の勾配を計算できるようにする
        Cx = torch.tensor(Cx, requires_grad=True, dtype=torch.float32)

        # GW distance
        p = torch.ones(n) / n
        q = torch.ones(m) / m
        P = compute_gw_transport(Cx, Cy, p, q)
        print(f"P.shape = {P.shape}")
        GW = compute_gromov_wasserstein_term(p, q, Cx, Cy, P)

        print(f"Epoch: {epoch}, GW distance: {GW}")

        # ----- BACKWARD -----
        # Implement backpropagation by hand

        GW.backward()

        # delta Cx
        L_Cx = Cx.grad.detach().numpy()
        print(f"L_Cx.shape = {L_Cx.shape}")

        # delta W
        dijkstra = DijkstraSS(alpha=alpha, distmat=W - lr * L_Cx + noise_W)
        map_Cx = dijkstra.dijkstra()
        L_W = Cx.detach().numpy() - map_Cx
        print(f"L_W.shape = {L_W.shape}")

        # delta ss
        idx = np.arange(n)
        A = np.abs(idx[:, None] - idx[None, :])

        L_ss = (np.zeros((n, n)) + alpha - A) * L_W
        print(f"L_ss.shape = {L_ss.shape}")

        # delta X
        X_new = X - lr * L_ss + noise_X
        # nussinov again
        sn = StochasticNussinov(X_new)
        sn.nussinov()
        ss_new = sn.structure_matrix()
        L_X = ss - ss_new
        L_X = L_W @ L_X

        # delta pi
        pi_new = pi - lr * L_X
        transition_scores_new = transition_scores - lr * L_X
        # kmm again
        kmm = KmerMarkovModel(kmer_size, pi_new + noise_pi,
                              transition_scores_new + noise_transition_scores)
        _, __, X = kmm.MAP_sequence(n)
        L_theta = X - X_new
        L_pi, L_transition_scores = L_theta


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
    Cy = torch.tensor(Cy, requires_grad=False, dtype=torch.float32)

    # parameters about x
    n = 30
    kmer_size = 3  # Length of the k-mer

    pi, transition_scores = train_model(
        kmer_size, n, m,  alpha, Cy, lr=1e-3, num_epochs=100)


if __name__ == "__main__":
    main()
