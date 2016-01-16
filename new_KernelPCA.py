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
import numba

# plt.interactive(True)

items = {}


def drawImage(image, shape=(16, 16), title="", dontDraw=False, ):
    image = image.astype(float)
    plt.imshow(np.reshape(image, shape), cmap=cm.Greys)
    plt.title(title)
    if not dontDraw:
        plt.show()
    else:
        plt.savefig("figures/%s.png" % title)


def add_noise(image, noise_type):
    if noise_type == 'speckle':
        image = random_noise(image, mode='s&p', salt_vs_pepper=0.5, amount=0.4)
    elif noise_type == 'gaussian':
        image = random_noise(image, mode='gaussian', var=0.5)
    return image


@numba.jit(nopython=True)
def K(x, y, N=1.0, std_dev=-999):
    c = 0.5
    if std_dev == -999:
        c = 0.5
    else:
        c = 2.0 * (std_dev * std_dev)
    # ret = -1.0 * spatial.distance.sqeuclidean(x, y)
    ret = -1.0 * np.sum(np.power(x-y, 2))
    ret = np.exp(ret / (c) / N
    return ret


def centered_kernel_matrix(X):
    num_data_points = X.shape[0]
    k_matrix = items['kernel matrix']
    cen_k_matrix = np.zeros(k_matrix.shape)
    for index_i in range(num_data_points):
        for index_j in range(num_data_points):
            cen_k_matrix[index_i, index_j] = \
                k_matrix[index_i, index_j] - \
                (np.sum(k_matrix[index_i, :]) / num_data_points) - \
                (np.sum(k_matrix[index_j, :]) / num_data_points) + \
                (np.sum(k_matrix) / (num_data_points ** 2))
    return cen_k_matrix


def kernel_matrix(X, centered=True, std_dev=-999):
    if 'kernel matrix' not in items:
        num_data_points = X.shape[0]
        k_matrix = np.zeros([X.shape[0], X.shape[0]])
        for i in range(num_data_points):
            for j in range(num_data_points):
                k_matrix[i, j] = K(X[i], X[j], num_data_points, std_dev)
        items['kernel matrix'] = k_matrix
    if centered:
        if 'centered kernel matrix' not in items:
            items['centered kernel matrix'] = centered_kernel_matrix(items['kernel matrix'])
        return items['centered kernel matrix']
    return items['kernel matrix']


def eigen_decomp(X):
    cen_k_matrix = kernel_matrix(X, centered=True, std_dev=-999)
    if 'eigen vectors' not in items:
        eig_val, eig_vec = linalg.eig(cen_k_matrix)
        idx = eig_val.argsort()[::-1]
        eig_val_sorted = eig_val[idx]
        eig_vec_sorted = eig_vec[:, idx]
        items['eigen vectors'] = eig_vec_sorted.astype(float)
    return items['eigen vectors']


def beta_test_point(X, X_test, num_comp, std_dev=-999):
    if 'beta' not in items:
        N = X.shape[0]
        beta = np.zeros(num_comp)
        eig_vec_sorted = eigen_decomp(X)

        for k in range(num_comp):
            for i in range(N):
                beta[k] += eig_vec_sorted[i][k] * K(X_test, X[i, :], 1, std_dev)
        items['beta'] = beta
    return items['beta']


def gamma(X, X_test, num_comp, std_dev=-999):
    if 'gamma' not in items:
        beta = beta_test_point(X, X_test, num_comp, std_dev)
        eigen_vectors = eigen_decomp(X)

        gammas = np.zeros(X.shape[0])
        for gamma_index in range(gammas.shape[0]):
            for comp in range(num_comp):
                gammas[gamma_index] = beta[comp] * eigen_vectors[gamma_index, comp]
        items['gamma'] = gammas
    return items['gamma']


def calculate_z(X, test_image, z, num_components, num_iterations, std_dev=-999):
    numerator = 0.0
    denominator = 0.0
    z = np.copy(test_image)

    if num_iterations == 0:  # Calculate gamma
        gamma(X, test_image, num_components, std_dev)
    gammas = items['gamma']


    # Calculate numerator and denominator
    for i in range(X.shape[0]):
        val = gammas[i] * K(z, X[i, :], 1, std_dev)
        numerator += val * X[i]
        denominator += val

    return numerator / denominator


def p_of_z(X, z, std_dev=-999):
    gammas = items['gamma']
    ret = 0.0
    for i in range(X.shape[0]):
        ret += gammas[i] * K(z, X[i, :], 1, std_dev)
    return ret  # TODO: add Omega value


def de_noise_image_multiple(X_train, X_test, num_components, num_iterations, std_dev=-999):
    results = np.zeros(X_test.shape)

    for index in range(X_test.shape[0]):
        print("****** starting on test image %s ******" % (index))
        current_pz = None
        z = np.copy(X_test[index])
        test_image = np.copy(X_test[index])
        for i in range(num_iterations):
            z = calculate_z(X_train, test_image, z, num_components, i, std_dev)
            pz = p_of_z(X_train, z, std_dev)
            if current_pz is None:
                current_pz = pz
            else:
                if current_pz <= pz or pz is np.nan:
                    break
        results[index, :] = z
    return results



def de_noise_image(X_train, X_test, num_components, num_iterations, std_dev=-999):
    current_pz = None
    z = np.copy(X_test)
    test_image = np.copy(X_test)
    for i in range(num_iterations):
        print("Starting on iteration %s" % i)
        z = calculate_z(X_train, test_image, z, num_components, i, std_dev)
        # pz = p_of_z(X_train, z, std_dev)
        # if current_pz is None:
        #     current_pz = pz
        # else:
        #     if current_pz <= pz or pz is np.nan:
        #         break
    return z



if __name__ == '__main__':
    train_count = 400
    test_data_point = 12
    num_components = 256
    num_iterations = 10

    usps_data, usps_target = datasets.usps_scikit()
    X_train, X_test, Y_train, Y_test = train_test_split(usps_data, usps_target, train_size=0.7, random_state=42)
    X = X_train[0:train_count]
    true_label = Y_test[test_data_point] - 1
    clean_image = X_test[test_data_point]
    drawImage(clean_image, title="%s initial" % (true_label), dontDraw=True)

    noisy_image = add_noise(clean_image, 'speckle')
    drawImage(noisy_image, title="%s noised" % (true_label), dontDraw=True)

    de_noised_image = de_noise_image(X, noisy_image, num_components, num_iterations)
    drawImage(de_noised_image, title="%s:%s recovered" % (true_label, num_components), dontDraw=True)