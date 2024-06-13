from modules.markov.kmer_mm import KmerMarkovModel
from modules.nussinov.stochastic_nussinov import StochasticNussinov
from modules.ss.internal_distance import  DijkstraSS

import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim

import ot
def compute_gw_distance(Cx, Cy, p, q, reg=0.1):
    """Compute the entropy-regularized Gromov-Wasserstein distance between two distance matrices."""
    gw_dist = ot.gromov.gromov_wasserstein2(Cx, Cy, p, q, 'square_loss', epsilon=reg)
    return gw_dist

class RNASeqModel(nn.Module):
    def __init__(self, kmer_size, num_states, alpha=0.5):
        super(RNASeqModel, self).__init__()
        self.kmer_model = KmerMarkovModel(kmer_size, num_states)  # Define this
        self.nussinov = StochasticNussinov()  # Define this
        self.dijkstra = DijkstraSS(alpha=alpha)

    def forward(self, initial_state, sequence_length, target_dist_matrix):
        # Generate sequence X from initial state
        X = self.kmer_model(initial_state, sequence_length)  # Implement this method

        # Apply Stochastic Nussinov to predict secondary structure
        secondary_structure = self.nussinov(X)  # Implement this method

        # Calculate the edge weight matrix from the secondary structure
        self.dijkstra.calculateDistmat(secondary_structure)

        # Compute minimum distance matrix using Dijkstra's algorithm
        dist_matrix = self.dijkstra.dijkstra()

        # Prepare uniform distribution over both matrices
        p = torch.ones(dist_matrix.shape[0]) / dist_matrix.shape[0]
        q = torch.ones(target_dist_matrix.shape[0]) / target_dist_matrix.shape[0]

        # Convert to numpy arrays for ot package
        Cx_np = dist_matrix.cpu().detach().numpy()
        Cy_np = target_dist_matrix.cpu().detach().numpy()
        p_np = p.cpu().detach().numpy()
        q_np = q.cpu().detach().numpy()

        # Compute GW distance
        gw_distance = compute_gw_distance(Cx_np, Cy_np, p_np, q_np)
        print(f"GW distance: {gw_distance}")
        

        return torch.tensor(gw_distance, requires_grad=True)  # Ensure gradient flow

# Assuming you define and initialize these objects somewhere
# Example training loop
def train_model():
    model = RNASeqModel(kmer_size=3, num_states=64, alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    target_dist_matrix = torch.randn(50, 50)  # This should be your actual target distance matrix
    initial_state = torch.randn(1, 64)  # Random initial state

    for _ in range(100):  # Number of training iterations
        optimizer.zero_grad()
        loss = model(initial_state, 50, target_dist_matrix)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
    return model

def test_model(model):
    pass

def main():
    model = train_model()
    
if __name__ == "__main__":
    main()