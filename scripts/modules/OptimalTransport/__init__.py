# def sinkhorn(p, q, C, epsilon=0.2, lambda_=1e-1, iter=100):
#     # initialize randomly
#     n, m = len(p), len(q)

#     p = torch.tensor(p, dtype=torch.float32, requires_grad=False)
#     q = torch.tensor(q, dtype=torch.float32, requires_grad=False)

#     n, m = len(p), len(q)
#     K = torch.exp(-C / epsilon)

#     u = torch.ones(n) / n

#     for t in range(iter):
#         print(K.T @ u)
#         # 要素ごとにわる
#         v = q / (K.T @ u)
#         u = p / (K @ v)

#     P_t = u.reshape(n, 1) * K * v.reshape(1, m)

#     return P_t
