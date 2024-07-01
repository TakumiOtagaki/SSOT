def parse_dot_bracket(dot_bracket):
    stack = []
    pairs = {}
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs

def create_distance_matrix(dot_bracket):
    n = len(dot_bracket)
    pairs = parse_dot_bracket(dot_bracket)
    B = [[1 if abs(i-j) <= 1 else 0 for j in range(n)] for i in range(n)]
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    A = [[1 if pairs.get(i) == j else 0 for j in range(n)] for i in range(n)]

    d = [[B[i][j] - I[i][j] + A[i][j] for j in range(n)] for i in range(n)]
    return d