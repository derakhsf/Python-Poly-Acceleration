import sys
import time
import numpy as np
from numba import njit, prange, float64, vectorize

# Array initialization
def init_array(m, n):
    alpha = 1.5
    beta = 1.2
    C = np.zeros((m, n), dtype=np.float64)
    A = np.zeros((m, m), dtype=np.float64)
    B = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        for j in range(n):
            C[i, j] = (i + j) % 100 / m
            B[i, j] = (n + i - j) % 100 / m

    for i in range(m):
        for j in range(i + 1):
            A[i, j] = (i + j) % 100 / m
        for j in range(i + 1, m):
            A[i, j] = -999  # regions of arrays that should not be used

    return alpha, beta, C, A, B

# Main computational kernel
def kernel_symm(m, n, alpha, beta, C, A, B):
    for i in range(m):
        temp2 = 0
        for j in range(n):
            for k in range(i):
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2

# Main computational kernel with loop unrolling
@njit(parallel=True)
def symm_unrolled(m, n, alpha, beta, C, A, B):
    for i in prange(m):
        temp2 = 0
        for j in range(n):
            for k in range(0, i):  # Unroll by 1
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2

# Vectorized inner loop
@vectorize(['float64(float64, float64, float64)'])
def vectorized_inner_loop(alpha, B_ij, A_ik):
    return alpha * B_ij * A_ik

# Vectorized symm function
@njit(parallel=True)
def symm_vectorized(m, n, alpha, beta, C, A, B):
    for i in prange(m):
        temp2 = 0
        for j in range(n):
            for k in range(i):  # Vectorized inner loop
                C[k, j] += vectorized_inner_loop(alpha, B[i, j], A[i, k])
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2

# Combined loop unrolling and vectorization
@njit(parallel=True)
def symm_combined(m, n, alpha, beta, C, A, B):
    for i in prange(m):
        temp2 = 0
        for j in range(n):
            for k in range(0, i, 4):  # Unroll by 4
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
                C[k + 1, j] += alpha * B[i, j] * A[i, k + 1]
                temp2 += B[k + 1, j] * A[i, k + 1]
                C[k + 2, j] += alpha * B[i, j] * A[i, k + 2]
                temp2 += B[k + 2, j] * A[i, k + 2]
                C[k + 3, j] += alpha * B[i, j] * A[i, k + 3]
                temp2 += B[k + 3, j] * A[i, k + 3]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2
'''
# Print the resulting array
def print_array(C):
    m, n = C.shape
    for i in range(m):
        for j in range(n):
            if (i * m + j) % 20 == 0:
                print()
            print(f"{C[i, j]:.2f}", end=" ")
'''

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python symm.py m n')
        sys.exit(1)

    m, n = map(int, sys.argv[1:3])

    alpha, beta, C, A, B = init_array(m, n)

    # Warm-up runs
    kernel_symm(m, n, alpha, beta, C, A, B)
    symm_unrolled(m, n, alpha, beta, C, A, B)
    symm_vectorized(m, n, alpha, beta, C, A, B)
    symm_combined(m, n, alpha, beta, C, A, B)

    num_runs = 20

    start = time.time()
    for _ in range(num_runs):
        kernel_symm(m, n, alpha, beta, C, A, B)
    end = time.time()
    naive_time = end - start

    start = time.time()
    for _ in range(num_runs):
        symm_unrolled(m, n, alpha, beta, C, A, B)
    end = time.time()
    unrolled_time = end - start

    start = time.time()
    for _ in range(num_runs):
        symm_vectorized(m, n, alpha, beta, C, A, B)
    end = time.time()
    vectorized_time = end - start

    start = time.time()
    for _ in range(num_runs):
        symm_combined(m, n, alpha, beta, C, A, B)
    end = time.time()
    combined_time = end - start

    print('naive time:      {}'.format(naive_time))
    print('unrolled time:   {}'.format(unrolled_time))
    print('vectorized time: {}'.format(vectorized_time))
    print('combined time:   {}'.format(combined_time))

