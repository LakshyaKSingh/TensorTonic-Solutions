import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A=np.array(A)
    rows=A.shape[0]
    columns=A.shape[1]
    Trans=np.zeros((columns,rows))
    for i in range(rows):
        for j in range(columns):
            Trans[j][i]=A[i][j]
    return Trans
    
