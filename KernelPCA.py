from __future__ import print_function
import numpy as np
from scipy import spatial
from sklearn import preprocessing
from numpy import linalg
from Data import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.util import random_noise
from sklearn.cross_validation import train_test_split
from time import time




class kernel_utils:
    def __init__(self, X):
        self.scaler = preprocessing.StandardScaler(with_std=False).fit(X)


def drawImage(image, shape=(16, 16), title=""):
    image = image.astype(float)
    plt.imshow(np.reshape(image, shape), cmap=cm.Greys)
    plt.title(title)
    plt.show()


def add_noise(image, noise_type):
    if noise_type == 'speckle':
        image = random_noise(image, mode='s&p', salt_vs_pepper=0.5, amount=0.4)
    elif noise_type == 'gaussian':
        image = random_noise(image, mode='gaussian', var=0.5)
    return image


def K(x1, x2, N):
    c = 1000  # TODO: change c to something different than 1.0
    ret = -1.0 * spatial.distance.sqeuclidean(x1, x2)
    return np.exp(ret / c) / N


def calculate_kernel_matrix(X):
    num_data_points = X.shape[0]
    k_matrix = np.zeros([X.shape[0], X.shape[0]])
    for i in range(num_data_points):
        for j in range(num_data_points):
            k_matrix[i, j] = K(X[i], X[j], num_data_points)
    # drawImage(k_matrix, shape=(256, 256))
    # quit()
    return k_matrix


def centered_kernel_matrix(X):
    num_data_points = X.shape[0]
    k_matrix = calculate_kernel_matrix(X)
    return k_matrix
    cen_k_matrix = np.zeros(k_matrix.shape)
    for index_i in range(num_data_points):
        for index_j in range(num_data_points):
            cen_k_matrix[index_i, index_j] = \
                k_matrix[index_i, index_j] - \
                (np.sum(k_matrix[index_i, :]) / num_data_points) - \
                (np.sum(k_matrix[index_j, :]) / num_data_points) + \
                (np.sum(k_matrix) / (num_data_points ** 2))
    return cen_k_matrix


def eigen_decomp(X):
    cen_k_matrix = centered_kernel_matrix(X)
    eig_val, eig_vec = linalg.eig(cen_k_matrix)
    idx = eig_val.argsort()[::-1]
    eig_val_sorted = eig_val[idx]
    eig_vec_sorted = eig_vec[:, idx]
    return eig_vec_sorted, cen_k_matrix


def beta_test_point(X, X_test, no_of_comp):
    # cen_X = centering_input_data(X)
    num_data_points = X.shape[0]
    beta = np.zeros((no_of_comp, 1))
    eig_vec_sorted, cen_k_matrix = eigen_decomp(X)
    for k in range(no_of_comp):
        for i in range(num_data_points):
            # beta[k] = beta[k] + eig_vec_sorted[i][k] * K(X_test, cen_X[i, :], 1)
            beta[k] = beta[k] + eig_vec_sorted[i][k] * K(X_test, X[i, :], 1)
    return beta, eig_vec_sorted


def gamma(X, X_test, no_of_comp, i):
    betas, eigen_vectors = beta_test_point(X, X_test, no_of_comp)
    gamma_val = 0.0
    for j in range(no_of_comp):
        gamma_val += betas[j] * eigen_vectors[j, i]
    return gamma_val


def p_of_z(X, X_test, no_of_comp):
    ret = 0.0
    for i in range(X.shape[0]):
        ret += gamma(X, X_test, no_of_comp, i) * K(X_test, X[i, :], 1)
    return ret  # TODO: add Omega value


def calculate_z(X, X_test, no_of_comp):
    numerator = 0.0
    denominator = 0.0
    z = np.copy(X_test)

    # Calculate numerator
    for i, line in enumerate(X):
        numerator += gamma(X, X_test, no_of_comp, i) * K(z, X[i, :], 1) * X[i]

    for i, line in enumerate(X):
        denominator += gamma(X, X_test, no_of_comp, i) * K(z, X[i, :], 1)

    #print("numerator:%s, denominator: %s" % (numerator, denominator))
    return numerator / denominator



# SIMPLE TEST

# X = np.array([[1., 2., 3., 6.],
#               [4., 5., 6., 8.],
#               [7., 8., 9., 3.],
#               [10., 11., 12., 3.]])
#
# X_test = [4, 13, 11, 1]
#
# for i in range(10):
#     X_test = calculate_z(X, X_test, 4)
#     pz = p_of_z(X, X_test, 4)
#     print("p(z) = %s" % pz)
#     print("X_test at iteration %s: %s" % (i, X_test))
# print(X_test)


# X, X_test = datasets.usps_resampled()
# X = np.vstack([np.array(digit) for digit in X])
# print(X.shape)
# quit()


# Prepare the data
train_count = 256
usps_data, usps_target = datasets.usps_scikit()
X_train, X_test, Y_train, Y_test = train_test_split(usps_data, usps_target, train_size=0.7, random_state=42)
X = X_train[0:train_count]
X_test = X_test[0]
true_label = Y_test[0]
drawImage(X_test, title="Before noise")
X_test = add_noise(X_test, 'speckle')
drawImage(X_test, title="After noise")


for i in range(3):
    start_time = time()
    print("iteration %s" % i)
    X_test = calculate_z(X, X_test, 1)
    pz = p_of_z(X, X_test, 1)
    #print("p(z) = %s" % pz)
    # print("X_test at iteration %s: %s" % (i, X_test))
    print("iteration %s took %.01f" % (i+1, time() - start_time))
#print(X_test)
drawImage(X_test, (16, 16), "%s: recovered" % true_label)

counter = 0
for val in X:
    if np.array_equal(val, X_test):
        counter += 1
print(counter)






# eigen_decomp(X)
# centered_X= centering_input_data(X)

# kernel_matrix = calculate_kernel_matrix(X)
# print(kernel_matrix)
