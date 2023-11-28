import sys
import time  
import numpy as np
from numba import njit, prange, vectorize, float64


# Array initialization
def init_array(n, m):
    A = np.zeros((n, m), dtype=np.float64)
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            A[i][j] = ((i * j + 1) % n) / n
    for i in range(n):
        for j in range(n):
            C[i][j] = ((i * j + 2) % m) / m
    return A, C



# Main computational kernel
def kernel_syrk(n, m, alpha, beta, C, A):
    for i in range(n):
        for j in range(i + 1):
            C[i][j] *= beta
        for k in range(m):
            for j in range(i + 1):
                C[i][j] += alpha * A[i][k] * A[j][k]
 
#loop_unrolled
def syrk_unrolled(n, m, alpha, beta, C, A):
    for i in range(n):
        for j in range(i + 1):
            C[i, j] *= beta
            for k in range(0, m, 4):  # Unroll by 4
                C[i, j] += alpha * A[i, k] * A[j, k]
                C[i, j] += alpha * A[i, k + 1] * A[j, k + 1]
                C[i, j] += alpha * A[i, k + 2] * A[j, k + 2]
                C[i, j] += alpha * A[i, k + 3] * A[j, k + 3]

                               
# Vectorized inner loop
@vectorize(['float64(float64, float64, float64)'])
def vectorized_inner_loop(alpha, A_ik, A_jk):
    return alpha * A_ik * A_jk

# Vectorized syrk function
@njit(parallel=True)
def syrk_vectorized(n, m, alpha, beta, C, A):
    for i in prange(n):
        for j in range(i + 1):
            C[i, j] *= beta
        for k in range(m):
            for j in range(i + 1):
                C[i, j] += vectorized_inner_loop(alpha, A[i, k], A[j, k])
            
#loop_unrolled_vectorized
@njit(parallel=True)
def syrk_combined_simple(n, m, alpha, beta, C, A):
    for i in prange(n):
        for j in range(i + 1):
            C[i, j] *= beta
            for k in range(m):
                C[i, j] += alpha * A[i, k] * A[j, k]
              
# Combined loop unrolling and vectorization
@njit(parallel=True)
def syrk_combined(n, m, alpha, beta, C, A):
    for i in prange(n):
        for j in range(i + 1):
            C[i, j] *= beta
            for k in range(0, m, 4):  # Unroll by 4 
                C[i, j] += alpha * A[i, k] * A[j, k]
                C[i, j] += alpha * A[i, k + 1] * A[j, k + 1]
                C[i, j] += alpha * A[i, k + 2] * A[j, k + 2]
                C[i, j] += alpha * A[i, k + 3] * A[j, k + 3]

# Start timer (not implemented here)
# polybench_start_instruments;




# Stop and print timer (not implemented here)
# polybench_stop_instruments;
# polybench_print_instruments;

# Print the resulting array
def print_array(C):
    for i in range(n):
        for j in range(n):
            if (i * n + j) % 20 == 0:
                print()
            print(f"{C[i][j]:.2f}", end=" ")

#print_array(C)



# main entry
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python syrk.py m n')
        sys.exit(1)
    
    alpha = 1.5
    beta = 1.2

    m, n = map(int, sys.argv[1:3])
    
    A, C = init_array(n, m)

    
    # Warm-up runs
    # Run kernel
    kernel_syrk(n, m, alpha, beta, C, A)
    syrk_unrolled(n, m, alpha, beta, C, A)
    syrk_vectorized(n, m, alpha, beta, C, A)
    syrk_combined_simple(n, m, alpha, beta, C, A)
    syrk_combined(n, m, alpha, beta, C, A)
    
    
    
    num_runs = 20

    start = time.time()
    for _ in range(num_runs):
        kernel_syrk(n, m, alpha, beta, C, A)
    end = time.time()
    naive_time = end - start

    #unrolled
    start = time.time()
    for _ in range(num_runs):
        syrk_unrolled(n, m, alpha, beta, C, A)
    end = time.time()
    unrolled_time = end - start


    #vectorized
    start = time.time()
    for _ in range(num_runs):
        syrk_vectorized(n, m, alpha, beta, C, A)
    end = time.time()
    vectorized_time = end - start
    
     #combined_simple
    start = time.time()
    for _ in range(num_runs):
        syrk_combined(n, m, alpha, beta, C, A)
    end = time.time()
    combined_simple_time = end - start
    
     #combined
    start = time.time()
    for _ in range(num_runs):
        syrk_combined(n, m, alpha, beta, C, A)
    end = time.time()
    combined_time = end - start
    

    
    

    print('naive time:      {}'.format(naive_time))
    print('unrolled time:   {}'.format(unrolled_time))
    print('vectorized time: {}'.format(vectorized_time))
    print('comb_simple time:{}'.format(combined_simple_time))
    print('combined time:   {}'.format(combined_time))

