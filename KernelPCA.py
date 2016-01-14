import numpy as np
from scipy import spatial
from sklearn import preprocessing
from numpy import linalg

def K(x1, x2, N):
    c = 1.0
    ret = -1.0 * spatial.distance.sqeuclidean(x1, x2) / N
    return np.exp(ret / c)

def centering_input_data(X):
    scaler=preprocessing.StandardScaler(with_std=False).fit(X)
    return scaler.transform(X)    

def calculate_kernel_matrix(X):
    num_data_points = X.shape[0]
    k_matrix = np.zeros([X.shape[0], X.shape[0]])
    for i, line_i in enumerate(X):
        for j, line_j in enumerate(X):            
            k_matrix[i, j] = K(line_i, line_j, num_data_points)
    return k_matrix

def centered_kernel_matrix(X):
    num_data_points=X.shape[0]
    cen_X= centering_input_data(X)
    k_matrix= calculate_kernel_matrix(cen_X)
    cen_k_matrix=np.zeros(k_matrix.shape)
    for i, line in enumerate(k_matrix):
        for j, _ in enumerate(line):
            cen_k_matrix[i,j]= k_matrix[i, j] - np.sum(k_matrix[i, :])/num_data_points - np.sum(k_matrix[j, :])/num_data_points + np.sum(k_matrix)/num_data_points**2
    return cen_k_matrix

def eigen_decomp(X):
    cen_k_matrix=centered_kernel_matrix(X)
    eig_val, eig_vec= linalg.eig(cen_k_matrix)    
    idx= eig_val.argsort()[::-1]
    eig_val_sorted = eig_val[idx]
    eig_vec_sorted = eig_vec[:, idx]
    print (eig_vec_sorted)
    return eig_vec_sorted, cen_k_matrix

def beta_test_point(X, X_test, no_of_comp):
    cen_X = centering_input_data(X)
    num_data_points = X.shape[0]
    beta = np.zeros((num_data_points,1))
    eig_vec_sorted, cen_k_matrix = eigen_decomp(X)
    for k in range(no_of_comp):
        for i in range(num_data_points):
            beta[k]=beta[k]+ eig_vec_sorted[k][i]*K(X_test, cen_X[i, :], 1)
    return beta 

X = np.array([[1, 2, 3, 6],
              [4, 5, 6, 8],
              [7, 8, 9, 3],
              [10, 11, 12, 3]])

X_test= [2, 6, 8, 1]
beta= beta_test_point(X, X_test, 4)
print beta
#eigen_decomp(X)
#centered_X= centering_input_data(X)
    
#kernel_matrix = calculate_kernel_matrix(X)
#print(kernel_matrix)
