import torch
import sys
import numpy as np
import re
from functools import partial
import pandas as pd
# from .load_params import parse_rna_params
# X は n times 4 matrix (torch)
LARGE = 1e6

""" 
for debug
"""

class RNAParams:
    def __init__(self):
        self.stack = dict()
        self.bulge = dict()
        self.hairpin = dict()
        self.internal = dict()
        self.ml_params = dict()
        self.stack_bases = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA', 'NN']
        self.limit_length = 30
        self.KT = 0.61632 # kcal/mol

def parse_rna_params(file_path):
    params = RNAParams()
    current_section = None
    stack_bases = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA', 'NN']

    section_patterns = {
        'stack': re.compile(r'#\s*stack$'),
        'bulge': re.compile(r'#\s*bulge$'),
        'hairpin': re.compile(r'#\s*hairpin$'),
        'internal': re.compile(r'#\s*interior$'),
        'ml_params': re.compile(r'#\s*ML_params$')
    }
    others_pattern = re.compile(r'#\s*')
    stack_bases = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA', 'NN']
    internal_length = 0 
    hairpin_length = 0
    bulge_length = 0
    stack_counter = 0


    with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            line = line.strip()
            # if /* ... */ comment, ignore: start with '/*' and end with '*/'
            if line.startswith('/*'):
                while not line.endswith('*/'):
                    line = next(file).strip()
                continue
            matched = False
            if not line or line.startswith('#'):
                # 末尾の /* ... */ コメントを除去
                line = line.split('/*')[0].strip()
                for section, pattern in section_patterns.items():
                    if pattern.search(line):
                        current_section = section
                        matched = True
                        break
                if not matched:
                    current_section = 'others'
                continue
            # print("current_section: ", current_section) 
            
            if current_section == "others":
                continue
            if current_section:
                # コメントを除去
                values = list(map(float, line.split("/*")[0].split()))
                # print("values: ", values)
                if current_section == 'stack':
                    left_base = stack_bases[stack_counter]
                    for i, value in enumerate(values):
                        right_base = stack_bases[i]
                        if left_base not in params.stack:
                            params.stack[left_base] = dict()
                        params.stack[left_base][right_base] = value
                    stack_counter += 1


                if current_section == 'bulge':
                    for value in values:
                        params.bulge[bulge_length] = value
                        bulge_length += 1

                elif current_section == 'hairpin':
                    for value in values:
                        params.hairpin[hairpin_length] = value
                        hairpin_length += 1
                elif current_section == 'internal':
                    for value in values:
                        params.internal[internal_length] = value
                        internal_length += 1
                elif current_section == 'ml_params':
                    params.ml_params["a"] = values[2] + values[5]
                    params.ml_params["b"] = values[3]
                    params.ml_params["c"] = values[4]   
    return params

def inf2large(prm, large=LARGE):
    for key in prm.bulge:
        if prm.bulge[key] == np.float64("inf"):
            prm.bulge[key] = large
    for key in prm.hairpin:
        if prm.hairpin[key] == np.float64("inf"):
            prm.hairpin[key] = large
    for key in prm.internal:
        if prm.internal[key] == np.float64("inf"):
            prm.internal[key] = large
    for key in prm.stack:
        for key2 in prm.stack[key]:
            if prm.stack[key][key2] == np.float64("inf"):
                prm.stack[key][key2] = large
    return prm

def scale_params(prm):
    # 1 の order にする
    scaler = 1e-3
    for key in prm.bulge:
        prm.bulge[key] = prm.bulge[key] * scaler
    for key in prm.hairpin:
        prm.hairpin[key] = prm.hairpin[key] * scaler
    for key in prm.internal:
        prm.internal[key] = prm.internal[key] * scaler
    for key in prm.stack:
        for key2 in prm.stack[key]:
            prm.stack[key][key2] = prm.stack[key][key2] * scaler
    prm.ml_params["a"] = prm.ml_params["a"] * scaler
    prm.ml_params["b"] = prm.ml_params["b"] * scaler
    prm.ml_params["c"] = prm.ml_params["c"] * scaler
    return prm

# X は文字列
def differential_canpair(x_i, x_j, T, gaussian_sigma):
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

def canpair(X, i, j, prm):
    return torch.tensor(X[i] + X[j] in prm.stack_bases)
    



def f1(i, j, X, prm):
    return torch.tensor(hairpin(i, j, X, prm))

def hairpin(i, j, X, prm):
    """
    i, j: int
    X: n times 4 matrix
    """
    d = j - i - 1
    if d > prm.limit_length:
        # return np.float64("inf") should be torch format
        return torch.tensor(LARGE)
    return prm.hairpin[d]

def f2(i, j, h, l, X, prm):
    # 例外処理
    if i > j or h > l:
        print(f"error: i >= j or h >= l in f2: {i}, {j}, {h}, {l}")
        sys.exit(1)
    if i + 1 == h and j - 1 == l:
        return torch.tensor(stack(i, j, X, prm))
    elif i + 1 == h:
        return torch.tensor(bulge(i, j, h, l, X, prm))
    elif j - 1 == l:
        return torch.tensor(bulge(i, j, h, l, X, prm))
    else:
        return torch.tensor(internal(i, j, h, l, X, prm))
    


def bulge(i, j, h, l, X, prm):
    if i + 1 == h:
        d = j - l - 1
        if d > prm.limit_length:
            return torch.tensor(LARGE)
        return prm.bulge[d]
    elif j - 1 == l:
        d = h - i - 1
        if d > prm.limit_length:
            return torch.tensor(LARGE)
        return prm.bulge[d]
    else:
        print("bulge error")
        sys.exit(1)

def internal(i, j, h, l, X, prm):
    """
    i, j, h, l: int
    X: n times 4 matrix
    """
    d1 = h - i - 1
    d2 = j - l - 1
    d = d1 + d2
    if d > prm.limit_length:
        return LARGE
    return prm.internal[d]
        

def stack(i, j, X, prm):
    """
    i, j: int
    X: n times 4 matrix
    """
    i_, j_ = i + 1, j - 1
    bp1 = X[i] + X[j]
    bp2 = X[i_] + X[j_]

    if bp1 not in prm.stack_bases or bp2 not in prm.stack_bases:
        return LARGE
    return prm.stack[bp1][bp2]

def mccaskill(X, prm):

    """
    X: n times 4 matrix
    T: temperature
    gaussian_sigma: variance of gaussian function
    """

    # initialization
    n = len(X)
    Z = [[torch.tensor(0.0, requires_grad=True) for _ in range(n)] for __ in range(n)]
    Z1 = [[torch.tensor(0.0, requires_grad=True) for _ in range(n)] for __ in range(n)]
    Zb = [[torch.tensor(0.0, requires_grad=True) for _ in range(n)] for __ in range(n)]
    Zm = [[torch.tensor(0.0, requires_grad=True) for _ in range(n)] for __ in range(n)]
    Zm1 = [[torch.tensor(0.0, requires_grad=True) for _ in range(n)] for __ in range(n)]

    a, b, c = prm.ml_params["a"], prm.ml_params["b"], prm.ml_params["c"]
    a, b, c = torch.tensor(a), torch.tensor(b), torch.tensor(c)
    kt = prm.KT
    # for all i, Z[i][i] = 1 and Z[i][i-1] = 1
    for i in range(n):
        Z[i][i] = Z[i][i] + 1
        if i > 0:
            Z[i][i-1] = Z[i][i-1] + 1
    


    # recursion
    for length in range(1, n):
        # print(f"hi, d: {length}")
        for i in range(n - length):


            # print(f"i: {i}")
            j = i + length
            # print("i, j ", i, j)
            if i < 0 or j > n: continue
            # Zb
            # print("f1(i, j, X, prm): ", f1(i, j, X, prm))
            Zb[i][j] = Zb[i][j] + torch.exp(- f1(i, j, X, prm) * canpair(X, i, j, prm) / kt)
            # print(f"Zb[{i}][{j}]: {Zb[i][j]}")
            for k in range(i + 1, j-1):
                for l in range(k+1, j):
                    Zb[i][j] = Zb[i][j] + Zb[k][l] * torch.exp(- f2(i, j, k, l, X, prm) * canpair(X, i, j, prm) * canpair(X, k, l, prm) / kt)
            for k in range(i + 1, j):
                Zb[i][j] = Zb[i][j] + Zm[i+1][k-1] * Zm1[k][j-1] * torch.exp(-a / kt)
            # print(f"Zb[{i}][{j}]: {Zb[i][j]}")

            # Z1
            for h in range(i, j + 1):
                Z1[i][j] = Z1[i][j] + Zb[i][h]
            # print(f"Z1[{i}][{j}]: {Z1[i][j]}")
            

            
            # Zm1
            for h in range(i, j + 1):
                Zm1[i][j] = Zb[i][h] * torch.exp(- c * (j - h) / kt)
            Zm1[i][j] = Zm1[i][j] * torch.exp(- b / kt)
            
            # Zm
            for h in range(i, j):
                Zm[i][j] = Zm[i][j] + (torch.exp( - c * ( h - i ) / kt) + Zm[i][h-1]) * Zm1[h][j]
            # sys.exit()


            # Z_ij
            Z[i][j] = Z[i][j] + 1.0
            for h in range(i, j+1):
                Z[i][j] = Z[i][j] + Z1[i][h-1] * Z1[h][j]
            # print(f"Z[{i}][{j}]: {Z[i][j]}")
        # if length == 1:
        #     print("d" , length)
        #     print(" a, b, c: ", a, b, c)
        #     # display(Z, Z1, Zb, Zm, Zm1)
        #     sys.exit()
    # sys.exit()
    return Z, Z1, Zb, Zm, Zm1


def outside(X, Z, Zb, Zm, prm):
    a, b, c = prm.ml_params["a"], prm.ml_params["b"], prm.ml_params["c"]
    a, b, c = torch.tensor(a), torch.tensor(b), torch.tensor(c)
    kt = prm.KT
    n = len(X)
    # initialization
    W = [[torch.tensor(0.0) for _ in range(n)] for __ in range(n)]
    for i in range(1, n):
        W[i][n-1] = W[i][n-1] + Z[0][i-1]
    for j in range(n-1):
        W[0][j] = W[0][j] + Z[j+1][n-1]

    # recursion
    for d in range(1, n):
        for i in range(n - d):
            j = d + i
            # if i == 0 or j == n-1: continue
            print("i, j: ", i, j)
            W[i][j] = W[i][j] + (Z[0][i-1] * Z[j+1][n-1] if i > 0 and j < n - 1 else torch.tensor(0.0))
            for h in range(i+1, j-1):
                for l in range(h+1, j):
                    W[i][j] += W[h][l] * torch.exp(-f2(i, j, h, l, X, prm) * canpair(X, i, j, prm) * canpair(X, h, l, prm) / kt)

            for h in range(i+1, j):
                for l in range(h+1, n):
                    term1 = Zm[h+1][i-1] * torch.exp(-(l-j-1)*c/kt) if h+1 < n and i-1 >= 0 else 0
                    term2 = Zm[j+1][l-1] * torch.exp(-(i-h-1)*c/kt) if j+1 < n and l-1 >= 0 else 0
                    term3 = Zm[h+1][i-1] * Zm[j+1][l-1] if h+1 < n and i-1 >= 0 and j+1 < n and l-1 >= 0 else 0
                    W[i][j] = W[h][l] * torch.exp(-(a+b)/kt) * (term1 + term2 + term3) + W[i][j]
    return W

def BPPM(Zb, W, Z):
    n = len(Zb)
    bppm = [[torch.tensor(0.0) for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(n):
            bppm[i][j] = Zb[i][j] * W[i][j] / Z[0][n-1]
    
    return bppm
    
def display(Z, Z1, Zb, Zm, Zm1):
    Z = [ [Z[i][j].item() for j in range(len(Z[i]))] for i in range(len(Z))]
    Z1 = [ [Z1[i][j].item() for j in range(len(Z1[i]))] for i in range(len(Z1))]
    Zb = [ [Zb[i][j].item() for j in range(len(Zb[i]))] for i in range(len(Zb))]
    Zm = [ [Zm[i][j].item() for j in range(len(Zm[i]))] for i in range(len(Zm))]
    Zm1 = [ [Zm1[i][j].item() for j in range(len(Zm1[i]))] for i in range(len(Zm1))]
    print("Z")
    print(pd.DataFrame(Z))

    print("Z1")
    print(pd.DataFrame(Z1))

    print("Zb")
    print(pd.DataFrame(Zb))

    print("Zm")
    print(pd.DataFrame(Zm))

    print("Zm1")
    print(pd.DataFrame(Zm1))

    

if __name__ == "__main__":
    print(torch.tensor(np.float64("inf")))
    prm = parse_rna_params("/large/otgk/SSOT/scripts/modules/mcCaskill/ViennaRNA/misc/rna_turner2004.par")
    prm = inf2large(prm)
    prm = scale_params(prm)
    x = "CCCCAAAAGGGG"
    Z, Z1, Zb, Zm, Zm1 = mccaskill(x, prm)
    W = outside(x, Z, Zb, Zm, prm)
    bppm = BPPM(Zb, W, Z)

    bppm = [ [bppm[i][j].item() for j in range(len(bppm[i]))] for i in range(len(bppm))]
    print(f"bppm ( x = {x} )")
    print(pd.DataFrame(bppm))
    sys.exit()

    print(Z[0][- 1].item())
    Z = [ [Z[i][j].item() for j in range(len(Z[i]))] for i in range(len(Z))]
    Z1 = [ [Z1[i][j].item() for j in range(len(Z1[i]))] for i in range(len(Z1))]
    Zb = [ [Zb[i][j].item() for j in range(len(Zb[i]))] for i in range(len(Zb))]
    Zm = [ [Zm[i][j].item() for j in range(len(Zm[i]))] for i in range(len(Zm))]
    Zm1 = [ [Zm1[i][j].item() for j in range(len(Zm1[i]))] for i in range(len(Zm1))]
    print("Z")
    print(pd.DataFrame(Z))

    print("Z1")
    print(pd.DataFrame(Z1))

    print("Zb")
    print(pd.DataFrame(Zb))

    print("Zm")
    print(pd.DataFrame(Zm))

    print("Zm1")
    print(pd.DataFrame(Zm1))