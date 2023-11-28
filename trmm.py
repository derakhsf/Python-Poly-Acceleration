import sys
import time  
import numpy as np
from numba import njit, prange,vectorize, float64

# Array initialization
def init_array(m, n):
    alpha = 1.5
    A = np.zeros((m, m), dtype=np.float64)
    B = np.zeros((m, n), dtype=np.float64)
    
    for i in range(m):
        for j in range(i):
            A[i, j] = ((i + j) % m) / m
        A[i, i] = 1.0
        for j in range(n):
            B[i, j] = ((n + (i - j)) % n) / n
    
    return alpha, A, B

# Main computational kernel
def kernel_trmm(m, n, alpha, A, B):
    for i in range(m):
        for j in range(n):
            for k in range(i + 1, m):
                B[i, j] += A[k, i] * B[k, j]
            B[i, j] = alpha * B[i, j]

# Main computational kernel with loop vectorization
@vectorize(['float64(float64, float64)'])
def vectorized_kernel(acc, a_b):
    return acc + a_b

@njit(parallel=True)
def kernel_trmm_vectorized(m, n, alpha, A, B):
    for i in prange(m):
        for j in range(n):
            acc = 0.0

            for k in range(i + 1, m):
                acc = vectorized_kernel(acc, A[k, i] * B[k, j])

            B[i, j] = alpha * (B[i, j] + acc)

# Main computational kernel with manual loop unrolling
@njit(parallel=True)
def kernel_trmm_unrolled(m, n, alpha, A, B):
    for i in prange(m):
        for j in range(n):
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            acc3 = 0.0

            for k in range(i + 1, m, 4):
                acc0 += A[k, i] * B[k, j]
                acc1 += A[k + 1, i] * B[k + 1, j]
                acc2 += A[k + 2, i] * B[k + 2, j]
                acc3 += A[k + 3, i] * B[k + 3, j]

            B[i, j] = alpha * (B[i, j] + acc0 + acc1 + acc2 + acc3)

# Main computational kernel with a combination of vectorization and unrolling
@njit(parallel=True)
def kernel_trmm_combined(m, n, alpha, A, B):
    for i in prange(m):
        for j in range(n):
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            acc3 = 0.0

            for k in range(i + 1, m, 4):
                acc0 = vectorized_kernel(acc0, A[k, i] * B[k, j])
                acc1 = vectorized_kernel(acc1, A[k + 1, i] * B[k + 1, j])
                acc2 = vectorized_kernel(acc2, A[k + 2, i] * B[k + 2, j])
                acc3 = vectorized_kernel(acc3, A[k + 3, i] * B[k + 3, j])

            B[i, j] = alpha * (B[i, j] + acc0 + acc1 + acc2 + acc3)

#

# Print the resulting array
def print_array(B):
    m, n = B.shape
    for i in range(m):
        for j in range(n):
            if (i * m + j) % 20 == 0:
                print()
            print(f"{B[i, j]:.2f}", end=" ")

if __name__ == '__main__':
    # Retrieve problem size.
    if len(sys.argv) != 3:
        print('Usage: python trmm.py m n ')
        sys.exit(1)
    m, n = map(int, sys.argv[1:3])

    # Initialize array(s).
    alpha, A, B = init_array(m, n)

    # Warm-up runs
    # Run kernel
    kernel_trmm(m, n, alpha, A, B)
    kernel_trmm_unrolled(m, n, alpha, A, B)
    kernel_trmm_vectorized(m, n, alpha, A, B)
    kernel_trmm_combined(m, n, alpha, A, B)
    
    
    
    num_runs = 20

    start = time.time()
    for _ in range(num_runs):
        kernel_trmm(m, n, alpha, A, B)
    end = time.time()
    naive_time = end - start

    #unrolled
    start = time.time()
    for _ in range(num_runs):
        kernel_trmm_unrolled(m, n, alpha, A, B)
    end = time.time()
    unrolled_time = end - start


    #vectorized
    start = time.time()
    for _ in range(num_runs):
        kernel_trmm_vectorized(m, n, alpha, A, B)
    end = time.time()
    vectorized_time = end - start
    

     #combined
    start = time.time()
    for _ in range(num_runs):
        kernel_trmm_combined(m, n, alpha, A, B)
    end = time.time()
    combined_time = end - start
    

    
    

    print('naive time:      {}'.format(naive_time))
    print('unrolled time:   {}'.format(unrolled_time))
    print('vectorized time: {}'.format(vectorized_time))
    print('combined time:   {}'.format(combined_time))

