import numpy as np
import math
import heapq # for dijkstra
import sys


class DijkstraSS:
    def __init__(self, alpha=0.5, distmat=None):
        self.alpha = alpha
        self.distmat = distmat
    
    def calculateDistmat(self, rna_ss_matrix): # excluding pseudoknot!
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
        print("dist in dijkstra end:")
        print(dist)
        return dist[end]

    def dijkstra(self):
        # Dijkstra's algorithm
        n = len(self.distmat)
        min_dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                min_dist_mat[i][j] = self.dijkstra_(i, j)
        return min_dist_mat
    


n = 10
rna_ss_matrix = np.zeros((n, n))
rna_ss_matrix[1,9] = 1
rna_ss_matrix[9,1] = 1
rna_ss_matrix[2,6] = 1
rna_ss_matrix[6,2] = 1


dijkstra_ss = DijkstraSS()
dijkstra_ss.calculateDistmat(rna_ss_matrix)
print("distmat")
print(dijkstra_ss.distmat)
min_distmat = dijkstra_ss.dijkstra()

print("min_distmat")
print(min_distmat)




                
