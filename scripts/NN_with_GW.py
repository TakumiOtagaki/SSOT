import math
from modules.markov.kmer_mm import KmerMarkovModel
from modules.nussinov.nussinov import nussinov, traceback
# from modules.nussinov.stochastic_nussinov import StochasticNussinov
from modules.nussinov.onehot_differential_nussinov import logsumexp_nussinov, traceback_MAP,  onehot_Relu_basepair, onehot_Relu_basepair_grad, grad_dp_X
from modules.nussinov.onehot_differential_nussinov import onehot_basepair, onehot_basepair_grad

from modules.ss.internal_distance import DijkstraSS, floyd_warshall, dijkstra_route
from modules.ss.util import bptuple2dotbracket
import numpy as np
import sys
import pandas as pd
import torch


from scipy.special import softmax

import ot
from modules.noise import SumOfGammaNoiseDistribution


DEBUG = True


# def check_convergence(f, g, epsilon, C, p, q, threshold=1e-7):
#     n, m = len(p), len(q)
#     P = torch.exp((-C + g.reshape(1, m) + f.reshape(n, 1)) / epsilon)
#     return torch.allclose(P.sum(0), q, atol=threshold) and torch.allclose(P.sum(1), p, atol=threshold)


# def log_sinkhorn(p, q, C, epsilon, iter):
#     epsilon = torch.tensor(epsilon, dtype=torch.float32, requires_grad=False)
#     n, m = len(p), len(q)
#     p = torch.tensor(ot.unif(n), dtype=torch.float32, requires_grad=False)
#     q = torch.tensor(ot.unif(m), dtype=torch.float32, requires_grad=False)
#     f = torch.zeros(n)
#     g = torch.zeros(m)

#     congergence = False
#     for t in range(iter):
#         f = -epsilon * \
#             torch.logsumexp((-C + g.reshape(1, m)) / epsilon,
#                             dim=1) + epsilon * torch.log(p)

#         # for debugging: all shape
#         # print(f"f.shape: {f.shape}, C.shape: {C.shape}, torch.logsumexp((-C + f.reshape(n, 1)) / epsilon,dim=0).shape: {torch.logsumexp((-C + f.reshape(n, 1)).T / epsilon,dim=0).shape}, epsilon * torch.log(q).shape: {(epsilon * torch.log(q)).shape}")

#         g = -epsilon * \
#             torch.logsumexp((-C + f.reshape(n, 1)) / epsilon,
#                             dim=0) + epsilon * torch.log(q)

#         if check_convergence(f, g, epsilon, C, p, q):
#             congergence = True
#             break

#     P = torch.exp((-C + g.reshape(1, m) + f.reshape(n, 1)) / epsilon)
#     if not congergence:
#         print("Not converged")
#     return P


# def compute_gromov_wasserstein_term(a, b, Cx, Cy, P):
#     """
#     Compute the Gromov-Wasserstein discrepancy term.

#     Args:
#     - ax, bx (numpy.ndarray): Weights for the source space X
#     - dX (numpy.ndarray): Distance matrix in the source space X
#     - ay, by (numpy.ndarray): Weights for the target space Y
#     - dY (numpy.ndarray): Distance matrix in the target space Y
#     - P (numpy.ndarray): Optimal transport plan between X and Y

#     Returns:
#     - float: Computed discrepancy term
#     """
#     # Calculate the weighted sum of squared distances for X
#     weighted_dX_squared = torch.sum(a[:, None] * a[None, :] * (Cx ** 2))

#     # Calculate the weighted sum of squared distances for Y
#     # weighted_dY_squared = torch.sum(b[:, None] * b[None, :] * (Cy ** 2))
#     weighted_dY_squared = torch.zeros(1)

#     cross_term = torch.einsum('ij,ik,jl,kl->', P, Cx, Cy, P)

#     # Combine terms
#     result = weighted_dX_squared + weighted_dY_squared - 2 * cross_term
#     return result


# def compute_gw_transport(Cx, Cy, p, q, epsilon, lambda_, max_iter, sinkhorn_iter):
#     # initialize randomly
#     n, m = len(p), len(q)
#     P_t = torch.ones(n, m) / (n * m)

#     for t in range(max_iter):
#         # C_t = -4 * np.dot(np.dot(Cx, P_t), Cy) # tensor に対応するように書き換える
#         C_t = -4 * torch.mm(torch.mm(Cx, P_t), Cy) + (epsilon - lambda_) * P_t
#         P_t = log_sinkhorn(p, q, C_t, epsilon=lambda_, iter=sinkhorn_iter)
#     return P_t


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
    return seq


def random_vector(n, dim):  # 尖った分布。x[i] のうちある d でだけ値が大きくあって欲しい
    X = torch.zeros(n, dim)
    for i in range(n):
        X[i] = torch.rand(dim)
        X[i] = torch.exp(- (X[i] - 1) ** 2 / 0.5)

    return X


def train_model(n, m, alpha, Cy, y, ss_y, lr, num_epochs, noise_scale, dim, T, ReluThreshold, gaussian_sigma, dijkstra_scaler):
    # --- pseudo code ---
    noise = SumOfGammaNoiseDistribution(k=3, nb_iterations=40)
    A = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            A[i, j] = torch.abs(torch.tensor(i - j))

    Bias = torch.zeros(n, n) * 3  # traceback の時にノイズによって非負の値が出るのを防止する

    # initialization
    # n times 4 matrix (one hot)
    X = torch.zeros(n, dim)
    for i in range(n):
        # random initialization
        X[i] = torch.rand(dim)
        # 1/0 に近づける
        X[i] = torch.exp(- (X[i] - 1) ** 2 / 0.1)
    seq0 = onehot2seq(X)
    print(X)

    for epoch in range(num_epochs):
        # 全ての grad の初期化

        print(
            f"----------------------------- Epoch: {epoch} --------------------------")
        # noise generation
        noise_W = noise.sample((n, n)) * noise_scale
        noise_theta = noise.sample((n, n, 3)) * noise_scale
        noise_phi = noise.sample((n, n, n)) * noise_scale

        # ----- FORWARD -----
        X = torch.tensor(X, requires_grad=True, dtype=torch.float32)
        X.grad = None

        # Stochastic Nussinov
        dp = logsumexp_nussinov(X, T, gaussian_sigma, ReluThreshold)
        dp = dp + Bias

        theta = torch.zeros(n, n, 3)
        phi = torch.zeros(n, n, n)
        for i in range(n-1):
            for j in range(i+1, n):
                theta[i][j][0] = theta[i][j][0] + dp[i+1][j]
                theta[i][j][1] = theta[i][j][1] + dp[i][j-1]
                # gaussian basepair
                # theta[i][j][2] = theta[i][j][2] + dp[i+1][j-1] + \
                #     onehot_basepair(
                #         X[i], X[j], T, gaussian_sigma) * (j - i >= 3)
                # Relu basepair
                theta[i][j][2] = theta[i][j][2] + dp[i+1][j-1] + \
                    onehot_Relu_basepair(
                        X[i], X[j], T, ReluThreshold) * (j - i >= 3)
                for k in range(i+1, j):
                    phi[i][j][k] = phi[i][j][k] + dp[i][k] + dp[k+1][j]

        # MAP 推定 on traceback
        # ↓ここに X が引数となっているのはまずいのでは？
        # ss_tuple = simple_traceback(dp + noise_dp, X, T, gaussian_sigma) # noise を足して負の項が出ないように。
        ss_tuple, z1, z2 = traceback_MAP(
            theta + noise_theta, phi + noise_phi, n)
        rna_ss_matrix = torch.zeros(n, n)
        rna_ss_matrix = rna_ss_matrix + z1[:, :, 2]

        ss_dot_bracket = bptuple2dotbracket(n, ss_tuple)
        print(f"seq 1st = {seq0}")
        print(f"seq now = {onehot2seq(X)}")
        print(f"ss : {ss_dot_bracket}")
        print(f"yss: {ss_y}")
        print(f"y  : {y}")

        # convert rna_ss_matrix to ss matrix
        print("rna_ss_matrix:", pd.DataFrame(rna_ss_matrix.detach().numpy()))
        W = calculateDistmat(rna_ss_matrix, alpha, A)
        print("W:", pd.DataFrame(W.detach().numpy()))
        W = W

        Cx = floyd_warshall(W + noise_W)

        # Cx を tensor にして、Loss (= GW) の勾配を計算できるようにする
        Cx = torch.tensor(Cx, requires_grad=True, dtype=torch.float32)

        Cy = Cy / Cy.max()
        Cx = Cx / Cx.max()

        print("Cx:", pd.DataFrame(Cx.detach().numpy()))
        print("Cy:", pd.DataFrame(Cy.detach().numpy()))

        # GW distance
        p = torch.tensor(ot.unif(n), dtype=torch.float32)
        q = torch.tensor(ot.unif(m), dtype=torch.float32)
        result = ot.gromov.entropic_gromov_wasserstein2(
            Cx, Cy, p, q, 'square_loss', log=True, verbose=True, epsilon= n * 5e-2, max_iter=1e14, symmetric=True)
        print("result = ", result)
        gw = result[-1]["gw_dist"]


        print(f"---------- GW distance: {gw} ----------")
        # sys.exit()

        # ----- BACKWARD -----
        # Implement backpropagation by hand

        gw.backward()
        print("Cx.grad:", Cx.grad)
        sys.exit()

        # delta Cx
        L_Cx = Cx.grad.detach().numpy()

        # delta W
        W_new = W - lr * L_Cx
        L_W = torch.zeros(n, n)

        print("dijkstra n^2 times")
        for i in range(n):
            for j in range(i+1, n):
                # if DEBUG:
                #     print(f"i = {i}, j = {j}")
                # O(n log n) * O(n^2) = O(n^3 log n)
                R_newij = dijkstra_route(W_new + noise_W, i, j)
                R_oldij = dijkstra_route(W + noise_W, i, j)
                L_W += 2 * (R_oldij - R_newij)

        # nussinov traceback の勾配を計算する
        # L_z1 を考える
        L_rna_ss_matrix = (alpha - A) * L_W
        L_z1 = torch.zeros(n, n, 3)
        L_z1[:, :, 2] = L_z1[:, :, 2] + L_rna_ss_matrix
        L_z2 = torch.zeros(n, n, n)

        new_theta = theta - lr * L_z1
        new_phi = phi - lr * L_z2

        # MAP
        new_ss_tuple, new_z1, new_z2 = traceback_MAP(new_theta + noise_theta,
                                                     new_phi + noise_phi, n)
        L_theta = z1 - new_z1
        L_phi = z2 - new_z2

        # もし L_theta[:,:, 2] == 0 だといかが全て 0 になるので、やり直す。ランダムな方向を X に加えてやり直す
        if torch.allclose(L_theta[:, :, 2], torch.zeros(n, n)):
            print("L_theta[:, :, 2] == 0; retry")
            X = X + random_vector(n, dim) * 10
            X = softmax(X.detach().numpy(), axis=1)
            continue

        L_dp = torch.zeros(n, n)
        L_X = torch.zeros(n, dim)
        for i in range(n-1):
            for j in range(i+1, n):
                L_dp[i][j] += L_theta[i-1][j+1][2] if i - \
                    1 >= 0 and j+1 < n else 0
                L_dp[i][j] += L_theta[i-1][j][0] if i-1 >= 0 else 0
                L_dp[i][j] += L_theta[i][j+1][1] if j+1 < n else 0
                for l in range(j+1, n):  # dp[i][j] が分岐の時に左側に出る場合
                    L_dp[i][j] += L_phi[i][l][j] if l < n else 0
                for l in range(i):  # dp[i][j] が分岐の時に右側に出る場合
                    L_dp[i][j] += L_phi[l][j][i-1] if l >= 0 else 0
        if DEBUG:
            # print("L_z1:", pd.DataFrame(L_z1[:, :, 2].detach().numpy()))
            print("theta:", pd.DataFrame(theta[:, :, 2].detach().numpy()))
            print("new_theta:", pd.DataFrame(
                new_theta[:, :, 2].detach().numpy()))
            # print("z1 = ", pd.DataFrame(z1[:, :, 2].detach().numpy()))
            # print("new_z1 = ", pd.DataFrame(new_z1[:, :, 2].detach().numpy()))
            print("z1 == new_z1:", torch.allclose(z1, new_z1))
        print("L_theta:", pd.DataFrame(L_theta[:, :, 2].detach().numpy()))
        # sys.exit()

        print("L_dp:", pd.DataFrame(L_dp.detach().numpy()))
        for i in range(n):
            # L_x[i, d] = sum_j(L_theta[i, j] * onehot_basepair(X[i], X[j], T, gaussian_sigma) * (j - i >= 3))
            for j in range(i+1, n):
                # L_X[i] += onehot_basepair_grad(
                #     X[i], X[j], T, gaussian_sigma) * L_theta[i][j][2] * (j - i >= 3)
                L_X[i] += onehot_Relu_basepair_grad(
                    X[i], X[j], T, ReluThreshold) * L_theta[i][j][2]

        # L_X
        ddpdX = grad_dp_X(dp, X, ReluThreshold, T, dim)
        print("ddpdX:", ddpdX)
        # ddpdX[n][dim][n][n]
        for t in range(n):
            for d in range(dim):
                for i in range(n):
                    for j in range(i+1, n):
                        L_X[t][d] += ddpdX[t][d][i][j] * L_dp[i][j]

        print("L_X:", pd.DataFrame(L_X.detach().numpy()))

        X = X - lr * L_X
        X = softmax(X.detach().numpy(), axis=1)
        print("X:", pd.DataFrame(X, dtype=float))
        # sys.exit()

        # print("L_X:", L_X)

        # もし X の各行の最大のものの最小値が 1 に非常に近いならば、収束したとみなし、終了する
        if np.min(np.max(X, axis=1)) > 0.5 and epoch > 2:
            break


def test():
    # print(torch.log(torch.tensor(0)))
    # sys.exit()
    pass


def main():

    alpha = 0.3
    # pre calculation for loss function
    y = "GGGGAAAAAACCCC"
    dp = nussinov(y)
    tuple_structure = traceback(dp, y, 0, len(y)-1)
    ss_y_string = bptuple2dotbracket(len(y), tuple_structure)
    ss_matrix_y = np.zeros((len(y), len(y)), dtype=int)
    for i, j in tuple_structure:
        ss_matrix_y[i][j] = 1
        ss_matrix_y[j][i] = 1
    y_dj = DijkstraSS(alpha=alpha)
    y_dj.calculateDistmat(ss_matrix_y)
    m = len(y)
    Cy = y_dj.dijkstra()
    # cy should be torch
    Cy = torch.tensor(Cy, dtype=torch.float32)

    # parameters about x
    n = m + 4
    dim = 4
    T = 100.0
    noise_scale = 1e-6
    num_epochs = 100
    dijkstra_scaler = 10.0
    Relu_Threshold = 0.7  # should be larger than 0.5, smaller than 1.0
    lr = torch.tensor(4e-2, requires_grad=False, dtype=torch.float32)
    gaussian_sigma = 0.1
    X = train_model(n, m,  alpha, Cy, y, ss_y_string, lr, num_epochs,
                    noise_scale, dim, T, Relu_Threshold, gaussian_sigma, dijkstra_scaler)


if __name__ == "__main__":
    test()
    main()
