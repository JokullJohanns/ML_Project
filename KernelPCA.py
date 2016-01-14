import numpy as np
from scipy import spatial


def K(x1, x2, N):
    c = 1.0
    ret = -1.0 * spatial.distance.sqeuclidean(x1, x2) / N
    return np.exp(ret / c)


def calculate_kernel_matrix(X):
    num_data_points = X.shape[0]
    k_matrix = np.zeros([X.shape[0], X.shape[0]])
    for i, line_i in enumerate(X):
        for j, line_j in enumerate(X):
            k_matrix[i, j] = K(line_i, line_j, num_data_points)
    return k_matrix


X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

kernel_matrix = calculate_kernel_matrix(X)
print(kernel_matrix)
