import numpy as np
import math
import heapq  # for dijkstra
import sys
import torch


class DijkstraSS:
    def __init__(self, alpha=0.5, distmat=None):
        self.alpha = alpha
        self.distmat = distmat

    def calculateDistmat(self, rna_ss_matrix):  # excluding pseudoknot!
        # rna secondary structure について、各塩基間の距離を計算する
        n = len(rna_ss_matrix)
        distmat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    distmat[i][j] = 0
                elif math.fabs(i-j) == 1:
                    distmat[i][j] = 1
                elif rna_ss_matrix[i][j] == 1:
                    distmat[i][j] = self.alpha
                    distmat[j][i] = self.alpha
                else:
                    distmat[i][j] = float('inf')
        self.distmat = distmat

    def dijkstra_(self, start, end):
        # Dijkstra's algorithm
        n = len(self.distmat)
        # using heap
        dist = [float('inf') for _ in range(n)]
        dist[start] = 0
        visited = [False for _ in range(n)]
        q = []
        heapq.heappush(q, (0, start))
        while len(q) > 0:
            d, u = heapq.heappop(q)
            if visited[u]:
                continue
            visited[u] = True

            for v in range(n):
                if visited[v]:
                    continue
                if dist[v] > dist[u] + self.distmat[u][v]:
                    dist[v] = dist[u] + self.distmat[u][v]
                    heapq.heappush(q, (dist[v], v))
        return dist[end]

    def dijkstra(self):
        # Dijkstra's algorithm
        if self.distmat is None:
            print("distmat is None")
            sys.exit()
        n = len(self.distmat)
        min_dist_mat = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                min_dist_mat[i][j] = self.dijkstra_(i, j)
        return min_dist_mat


def dijkstra_route(W, s, g):
    # return the best root matrix R: (s to g)
    # R_ij = 1 if best routes contains edge (i to j)
    # use heap

    n = len(W)
    dist = [float('inf') for _ in range(n)]
    dist[s] = 0
    visited = [False for _ in range(n)]
    q = []
    heapq.heappush(q, (0, s))
    while len(q) > 0:
        d, u = heapq.heappop(q)
        if visited[u]:
            continue
        visited[u] = True

        for v in range(n):
            if visited[v]:
                continue
            if dist[v] > dist[u] + W[u][v]:
                dist[v] = dist[u] + W[u][v]
                heapq.heappush(q, (dist[v], v))

    # backtracking
    R = torch.zeros((n, n))
    u = g
    while u != s:
        for v in range(n):
            if dist[u] == dist[v] + W[v][u]:
                R[v][u] = R[v][u] + 1
                u = v
                break
    return R


def floyd_warshall(W):
    # return the shortest distance matrix D
    # D_ij is the shortest distance from i to j
    n = len(W)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = W[i][j]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i][j] = min(D[i][j], D[i][k] + D[k][j])
    return D


if __name__ == "__main__":

    n = 10
    rna_ss_matrix = np.zeros((n, n))
    rna_ss_matrix[1, 9] = 1
    rna_ss_matrix[9, 1] = 1
    rna_ss_matrix[2, 6] = 1
    rna_ss_matrix[6, 2] = 1

    dijkstra_ss = DijkstraSS()
    dijkstra_ss.calculateDistmat(rna_ss_matrix)
    print("distmat")
    print(dijkstra_ss.distmat)
    min_distmat = dijkstra_ss.dijkstra()

    print("min_distmat")
    print(min_distmat)
