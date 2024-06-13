import numpy as np
import random
import itertools
import sys
import numpy as np
import itertools


class KmerMarkovModel:
    def __init__(self, k, pi, transition_scores):
        self.k = k
        if k > 5:
            print("Warning: The number of states grows exponentially with k. Use with caution.")
            input("Press Enter to continue...")
        self.states = self.generate_states()
        self.num_states = len(self.states)
        self.pi = pi
        self.transition_scores = transition_scores

        self.base_mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        
    def generate_states(self):
        """ Generate all possible k-mer states """
        bases = ['A', 'U', 'G', 'C']
        return [''.join(seq) for seq in itertools.product(bases, repeat=self.k)]
    
    # def state_to_index(self, state):
    #     """ Convert a base sequence to a quaternary number """
    #     base_mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}


    #     return sum(base_mapping[base] * (4 ** i) for i, base in enumerate(reversed(state)))

    def MAP_sequence(self, n):
        """ Generate an optimal sequence of length n using dynamic programming """
        if n < self.k:
            raise ValueError("n must be at least k to form a complete sequence.")
        
        # Initialize DP table
        dp = np.full((self.num_states, n - self.k + 1), -np.inf)
        # dp[:, 0] = 0  # Starting from any state at position 0 
        dp[:, 0] = self.pi
        # dp[i, j] means the score of the optimal sequence ending in state i at position j
        
        # Backtrack table
        backtrack = np.zeros((self.num_states, n - self.k + 1), dtype=int)

        # Fill DP table
        for j in range(1, n - self.k + 1): # j は配列長に対応
            for i in range(self.num_states):
                for b in range(4):  # Loop over possible new bases
                    new_state_idx = (i * 4 + b) % self.num_states
                    score = dp[i, j-1] + self.transition_scores[i, b]
                    if score > dp[new_state_idx, j]:
                        dp[new_state_idx, j] = score
                        backtrack[new_state_idx, j] = i
            

        # Find the optimal path
        end_state = np.argmax(dp[:, -1])
        optimal_sequence = [self.states[end_state]]
        optimal_seq_matrix = np.zeros((n, 4), dtype=float)



        for j in range(n - self.k, -1, -1):
            end_state = backtrack[end_state, j]
            # print("end_state:", self.states[end_state])
            optimal_sequence.append(self.states[end_state][0])
            for b in range(4):
                # j-1 まではそのままで、j 番目に b が来た時のそこまでの score を書く
                optimal_seq_matrix[j, b] = dp[end_state, j] + self.transition_scores[end_state, b]
                # print("optimal_seq_matrix[j, b]:", optimal_seq_matrix[j, b])
            optimal_seq_matrix[j] = optimal_seq_matrix[j] / np.sum(optimal_seq_matrix[j])

        # 最後の k-1 列は、後ろから k 番目のものを写す
        for j in range(n - self.k, n):
            optimal_seq_matrix[j] = optimal_seq_matrix[n - self.k]
        return ''.join(reversed(optimal_sequence)), dp[end_state, -1], optimal_seq_matrix
        

# Example usage
k = 3  # Length of the k-mer
# seed
np.random.seed(6)
pi = np.random.rand(4 ** k)  # Random initial probabilities
transition_scores = np.random.rand(4 ** k, 4)  # Random transition scores
print("Initial probabilities:", pi)
print("Transition scores:", transition_scores)
# sys.exit()

model = KmerMarkovModel(k, pi, transition_scores)
sequence_length = 50
generated_sequence, score, optimal_seq_matrix = model.MAP_sequence(sequence_length)
print("Generated sequence:", generated_sequence)
print("Optimal score:", score)
print("Optimal sequence matrix:", optimal_seq_matrix)