# rna seq matrix:
# 4 x n
# rna_seq_matrix = [ \
# [1, 0, 1, 0, 0], # A を表す行
# [0, 1, 0, 0, 0], # U を表す行
# [0, 0, 0, 0, 1], # G を表す行
# [0, 0, 0, 1, 0]  # C を表す行
#  ]
# このときrna_seq = "AUACG" となる。


# from math import fabs
# def can_pair(i, j, rna_seq):
#     pair = set((rna_seq[i], rna_seq[j]))
#     valid_pairs = [set('AU'), set('GU'), set('GC')]
#     return any(pair == vp for vp in valid_pairs) * (fabs(i-j) > 3)

import sys


def validate_input(rna_seq_matrix):
    # もし rna_seq のある列の合計値が 0 のとき、エラー出す
    # もし rna_seq のある値が 0 未満の時もエラーを出す
    pass


class StochasticNussinov:
    def __init__(self, rna_seq_matrix):
        self.rna_seq_matrix = rna_seq_matrix
        self.rna_partial_sum = [sum(rna_seq_matrix[i])
                                for i in range(len(rna_seq_matrix))]
        self.dp = [[0] * len(rna_seq_matrix)
                   for _ in range(len(rna_seq_matrix))]
        self.strcture = set()
        self.n = len(rna_seq_matrix)
        self.num_type = len(rna_seq_matrix[0])

    def expec_pair(self, i, j):
        if max(i, j) - min(i, j) < 3:
            return 0

        def can_pair_idx(s, t):
            # s, res = (0, A), (1, U), (2, G), (3, C)
            return (s, t) in [(0, 1), (1, 0), (2, 3), (3, 2)]
        ret = 0
        for s in range(4):
            for t in range(4):
                ret += self.rna_seq_matrix[i][s] * \
                    self.rna_seq_matrix[j][t] * can_pair_idx(s, t)
        return ret / (self.rna_partial_sum[j] * self.rna_partial_sum[i])

    def nussinov(self):
        # Fill the DP table
        for length in range(4, self.n + 1):  # Minimum loop length condition
            for i in range(self.n - length + 1):
                j = i + length - 1
                # if can_pair(i, j, rna_seq):
                #     dp[i][j] = max(dp[i][j], dp[i+1][j-1] + 1)
                self.dp[i][j] = max(
                    [self.dp[i][j],
                     self.dp[i+1][j],
                     self.dp[i][j-1],
                     self.dp[i+1][j-1] + self.expec_pair(i, j)]
                    + [self.dp[i][k] + self.dp[k+1][j] for k in range(i+1, j)]
                )
        self.traceback()

    # いずれ traceback の dp に noise が加わるので注意する。

    def traceback(self):
        self.structure = self.traceback_(0, self.n-1)

    def nearly_equal(self, a, b, epsilon=1e-6):
        return abs(a - b) < epsilon

    def traceback_(self, i, j,  structure=set()):
        if i >= j:
            return structure
        # elif self.dp[i][j] == self.dp[i+1][j]:
        elif self.nearly_equal(self.dp[i][j], self.dp[i+1][j]):
            return self.traceback_(i+1, j, structure)
        # elif self.dp[i][j] == self.dp[i][j-1]:
        elif self.nearly_equal(self.dp[i][j], self.dp[i][j-1]):
            return self.traceback_(i, j-1, structure)
        # elif self.dp[i][j] == self.dp[i+1][j-1] + self.expec_pair(i, j):
        elif self.nearly_equal(self.dp[i][j], self.dp[i+1][j-1] + self.expec_pair(i, j)):
            structure.add((i, j))
            return self.traceback_(i+1, j-1, structure)
        else:
            for k in range(i+1, j):
                # if self.dp[i][j] == self.dp[i][k] + self.dp[k+1][j]:
                if self.nearly_equal(self.dp[i][j], self.dp[i][k] + self.dp[k+1][j]):
                    return self.traceback_(i, k,  structure) | self.traceback_(k+1, j, structure)

    def structure_matrix(self):
        ss_matrix = [[0] * self.n for _ in range(self.n)]
        for (i, j) in self.structure:
            ss_matrix[i][j] = 1
            ss_matrix[j][i] = 1
        return ss_matrix


class Rnaseq:
    def __init__(self, rna_seq):
        self.rna_seq = rna_seq
        self.rna_seq_matrix = self.convert_seq2matrix_()

    def convert_seq2matrix_(self):
        # Create an Nx4 matrix where each row corresponds to a base and has one-hot encoding
        rna_seq_matrix = [
            [1 if base == 'A' else 0,
             1 if base == 'U' else 0,
             1 if base == 'G' else 0,
             1 if base == 'C' else 0] for base in self.rna_seq
        ]
        return rna_seq_matrix

    def add_noise(self, noise_level=0.1):
        import random
        for i in range(len(self.rna_seq_matrix)):
            for j in range(len(self.rna_seq_matrix[i])):
                r = random.random() * noise_level
                self.rna_seq_matrix[i][j] += r
        self.rna_seq_matrix = [[max(0, min(1, x)) for x in row]
                               for row in self.rna_seq_matrix]


if __name__ == "__main__":
    # Example usage
    rnaseq = Rnaseq("AACCCCUUAAAAGGGGCCCC")
    rnaseq.add_noise(0.2)
    rna_seq_matrix = rnaseq.rna_seq_matrix
    rna_seq = rnaseq.rna_seq

    stochastic_nussinov = StochasticNussinov(rna_seq_matrix)
    stochastic_nussinov.nussinov()
    stochastic_nussinov.traceback()
    structure = stochastic_nussinov.structure

    print("Pairs:", structure)
    # dot bracket
    dot_bracket = ['.' for _ in range(len(rna_seq))]
    for (i, j) in structure:
        dot_bracket[i] = '('
        dot_bracket[j] = ')'

    print("RNA sequence:\n\t", rna_seq)
    print("Dot bracket:\n\t", ''.join(dot_bracket))

    structure_matrix = stochastic_nussinov.structure_matrix()
    for i in range(len(rna_seq)):
        print(structure_matrix[i])
