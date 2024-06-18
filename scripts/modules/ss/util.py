
def bptuple2dotbracket(n, bp_tuple):
    # bp_tuple: (i, j) の tuple の set
    # n: 長さ
    ret = ['.'] * n
    for i, j in bp_tuple:
        ret[i] = '('
        ret[j] = ')'
    return ''.join(ret)
