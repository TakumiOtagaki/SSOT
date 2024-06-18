import torch
import sys
# X は n times 4 matrix (torch)


def onehot_basepair(x_i, x_j, T=3.0):
    # xi, xj はどちらも 0/1 の四次元ベクトル。A, U, G, C の順番に並んでいる。
    # if (xi, xj) is (A, U), (U, A), (G, C), (C, G), (U, G), (G, U): return 1
    # これを 内積で表現する（微分可能にするために）
    A_U = x_i[0] * x_j[1] + x_i[1] * x_j[0]  # 1
    # print("A_U:", A_U)
    G_C = x_i[2] * x_j[3] + x_i[3] * x_j[2]  # 2
    # print("G_C:", G_C)
    U_G = x_i[1] * x_j[2] + x_i[2] * x_j[1]  # 3
    # print("U_G:", U_G)

    return ((A_U + G_C + U_G) / (x_i.sum() * x_j.sum())) * T


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


def logsumexp_nussinov(X, T=3.0):
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
                    dp[i+1][j-1] +
                    onehot_basepair(X[i], X[j], T)  # 3
                ]
                    + [dp[i][k] + dp[k+1][j] for k in range(i+1, j)]), dim=0
            )

            mask = torch.zeros(n, n)
            mask[i, j] = 1
            dp = dp + mask * dp_ij

    return dp


def calc_intermediate_params(dp, X, T=3.0):
    n = dp.size(0)
    theta = torch.zeros(n, n, 3)  # i, j to i', j'
    phi = torch.zeros(n, n, n)  # i, j to i, k and k+1, j

    # i, j, i+1, j と i,j,i,j-1 と i,j,i+1,j-1 だけ theta の値を dp[*,*] に変更
    for i, j in [(i, j) for i in range(j) for j in range(n)]:
        # index error に注意して書く↓
        theta[i, j, 0] = dp[i+1][j]
        theta[i, j, 1] = dp[i][j-1]
        theta[i, j, 2] = dp[i+1][j-1] + \
            onehot_basepair(X[i], X[j], T)
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


def simple_traceback(dp, X, T):
    n = X.size(0)
    structure = set()  # tuple of (i, j)

    def traceback_(i, j):
        try:
            nonlocal structure
            if i >= j:
                return
            choices = torch.stack([
                dp[i+1][j],
                dp[i][j-1],
                dp[i+1][j-1] + onehot_basepair(X[i], X[j], T),
            ] + [dp[i][k] + dp[k+1][j] for k in range(i+1, j)]).unsqueeze(0)
            max_index = torch.argmax(choices).item()
            if max_index == 2:
                structure.add((i, j))
                traceback_(i+1, j-1)
            elif max_index == 0:
                traceback_(i+1, j)
            elif max_index == 1:
                traceback_(i, j-1)
            else:
                k = max_index - 3 + i
                traceback_(i, k)
                traceback_(k+1, j)
        except Exception as e:
            print("i, j:", i, j)
            print("ERROR!, ", e)
            sys.exit()
            raise e

    traceback_(0, n-1)
    return structure


if __name__ == "__main__":
    x1, x2 = "A", "A"
    s1 = res2onehot(x1)
    s2 = res2onehot(x2)
    print(onehot_basepair(s1, s2))
