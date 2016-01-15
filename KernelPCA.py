import numpy as np
from scipy import spatial
from sklearn import preprocessing
from numpy import linalg
from Data import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.util import random_noise
from sklearn.cross_validation import train_test_split


def drawImage(image, shape):
    plt.imshow(np.reshape(image, shape), cmap=cm.Greys)
    plt.show()


def add_noise(image, noise_type):
    if noise_type == 'speckle':
        image = random_noise(image, mode='s&p', salt_vs_pepper=0.5, amount=0.4)
    elif noise_type == 'gaussian':
        image = random_noise(image, mode='gaussian', var=0.5)
    return image


class KernelPCA:
    def __init__(self, X, c=1.0):
        self.X = X
        self.c = c

    def K(self, x1, x2, N=1.0):
        ret = -1.0 * spatial.distance.sqeuclidean(x1, x2) / N
        return np.exp(ret / self.c)




def K(x1, x2, N):
    c = 1.0
    ret = -1.0 * spatial.distance.sqeuclidean(x1, x2) / N
    return np.exp(ret / c)


def centering_input_data(X):
    scaler = preprocessing.StandardScaler(with_std=False).fit(X)
    return scaler.transform(X)


def calculate_kernel_matrix(X):
    num_data_points = X.shape[0]
    k_matrix = np.zeros([X.shape[0], X.shape[0]])
    for i, line_i in enumerate(X):
        for j, line_j in enumerate(X):
            k_matrix[i, j] = K(line_i, line_j, num_data_points)
    return k_matrix


def centered_kernel_matrix(X):
    num_data_points = X.shape[0]
    cen_X = centering_input_data(X)
    k_matrix = calculate_kernel_matrix(cen_X)
    cen_k_matrix = np.zeros(k_matrix.shape)
    for i, line in enumerate(k_matrix):
        for j, _ in enumerate(line):
            cen_k_matrix[i, j] = k_matrix[i, j] - np.sum(k_matrix[i, :]) / num_data_points - np.sum(
                    k_matrix[j, :]) / num_data_points + np.sum(k_matrix) / num_data_points ** 2
    return cen_k_matrix


def eigen_decomp(X):
    cen_k_matrix = centered_kernel_matrix(X)
    eig_val, eig_vec = linalg.eig(cen_k_matrix)
    idx = eig_val.argsort()[::-1]
    eig_val_sorted = eig_val[idx]
    eig_vec_sorted = eig_vec[:, idx]
    #print(eig_vec_sorted)
    return eig_vec_sorted, cen_k_matrix


def beta_test_point(X, X_test, no_of_comp):
    cen_X = centering_input_data(X)
    num_data_points = X.shape[0]
    beta = np.zeros((no_of_comp, 1))
    eig_vec_sorted, cen_k_matrix = eigen_decomp(X)
    for k in range(no_of_comp):
        for i in range(num_data_points):
            beta[k] = beta[k] + eig_vec_sorted[i][k] * K(X_test, cen_X[i, :], 1)
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

usps_data = datasets.usps()
data = []
for i in range(10):
    data.append(usps_data[i][0:10])
X = np.vstack(data)
X = X.astype(float)
print(X.shape)

X_test = usps_data[0][44]
#drawImage(X_test, (16, 16))
X_test = add_noise(X_test, 'gaussian')
#drawImage(X_test, (16, 16))
X_test = X_test.astype(float)

print(X_test)
quit()

for i in range(3):
    print("here")
    X_test = calculate_z(X, X_test, 1)
    pz = p_of_z(X, X_test, 1)
    print("p(z) = %s" % pz)
    # print("X_test at iteration %s: %s" % (i, X_test))
#rint(X_test)
drawImage(X_test, (16, 16))

counter = 0
for val in X:
    if np.array_equal(val, X_test):
        counter += 1
print(counter)






# eigen_decomp(X)
# centered_X= centering_input_data(X)

# kernel_matrix = calculate_kernel_matrix(X)
# print(kernel_matrix)
