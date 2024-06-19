from math import fabs


def can_pair(i, j, rna_seq):
    pair = set((rna_seq[i], rna_seq[j]))
    valid_pairs = [set('AU'), set('GU'), set('GC')]
    return any(pair == vp for vp in valid_pairs) * (fabs(i-j) > 3)


def nussinov(rna_seq):
    n = len(rna_seq)
    # Initialize the DP table
    dp = [[0]*n for _ in range(n)]

    # Fill the DP table
    for length in range(4, n+1):  # Minimum loop length condition
        for i in range(n-length+1):
            j = i + length - 1
            if can_pair(i, j, rna_seq):
                dp[i][j] = max(dp[i][j], dp[i+1][j-1] + 1)
            dp[i][j] = max(dp[i][j], dp[i+1][j], dp[i][j-1])
            for k in range(i+1, j):
                dp[i][j] = max(dp[i][j], dp[i][k] + dp[k+1][j])

    return dp


def traceback(dp, rna_seq, i, j, structure=set()):
    if i >= j:
        return structure
    elif dp[i][j] == dp[i+1][j]:
        return traceback(dp, rna_seq, i+1, j, structure)
    elif dp[i][j] == dp[i][j-1]:
        return traceback(dp, rna_seq, i, j-1, structure)
    elif dp[i][j] == dp[i+1][j-1] + 1 and can_pair(i, j, rna_seq):
        structure.add((i, j))
        return traceback(dp, rna_seq, i+1, j-1, structure)
    else:
        for k in range(i+1, j):
            if dp[i][j] == dp[i][k] + dp[k+1][j]:
                return traceback(dp, rna_seq, i, k, structure) | traceback(dp, rna_seq, k+1, j, structure)


if __name__ == "__main__":
    # Example usage
    rna_sequence = "AAUCUUAUCAAUUAAUUUGAAUACAGAAGA"
    dp = nussinov(rna_sequence)
    tuple_structure = traceback(dp, rna_sequence, 0, len(rna_sequence)-1)
    print("Pairs:", tuple_structure)
    # dot bracket
    dot_bracket = ['.' for _ in range(len(rna_sequence))]
    for (i, j) in tuple_structure:
        dot_bracket[i] = '('
        dot_bracket[j] = ')'

    print("RNA sequence:\n\t", rna_sequence)
    print("Dot bracket:\n\t", ''.join(dot_bracket))
    print("num bp:", len(tuple_structure))
