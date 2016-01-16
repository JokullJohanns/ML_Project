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


def K(x, y, N=1.0):
    c = 0.5
    ret = -1.0 * spatial.distance.sqeuclidean(x, y)
    ret = np.exp(ret / c) / N
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


def kernel_matrix(X, centered=True):
    if 'kernel matrix' not in items:
        num_data_points = X.shape[0]
        k_matrix = np.zeros([X.shape[0], X.shape[0]])
        for i in range(num_data_points):
            for j in range(num_data_points):
                k_matrix[i, j] = K(X[i], X[j], num_data_points)
        items['kernel matrix'] = k_matrix
    if centered:
        if 'centered kernel matrix' not in items:
            items['centered kernel matrix'] = centered_kernel_matrix(items['kernel matrix'])
        return items['centered kernel matrix']
    return items['kernel matrix']


def eigen_decomp(X):
    cen_k_matrix = kernel_matrix(X, centered=True)
    if 'eigen vectors' not in items:
        eig_val, eig_vec = linalg.eig(cen_k_matrix)
        idx = eig_val.argsort()[::-1]
        eig_val_sorted = eig_val[idx]
        eig_vec_sorted = eig_vec[:, idx]
        items['eigen vectors'] = eig_vec_sorted.astype(float)
    return items['eigen vectors']


def beta_test_point(X, X_test, no_of_comp):
    num_data_points = X.shape[0]
    beta = np.zeros((no_of_comp, 1))
    eig_vec_sorted = eigen_decomp(X)
    for k in range(no_of_comp):
        for i in range(num_data_points):
            beta[k] += eig_vec_sorted[i][k] * K(X_test, X[i, :], 1)
    return beta, eig_vec_sorted


def gamma(X, X_test, no_of_comp, i):
    betas, eigen_vectors = beta_test_point(X, X_test, no_of_comp)
    gamma_val = 0.0
    for j in range(no_of_comp):
        gamma_val += betas[j] * eigen_vectors[j, i]
    return gamma_val


def calculate_z(X, X_test, no_of_comp):
    numerator = 0.0
    denominator = 0.0
    z = np.copy(X_test)
    # if iteration == 0:
    #     z = np.average(X, 0)
    # else:
    #     z = np.copy(X_test)

    # Calculate numerator
    for i in range(X.shape[0]):
        numerator += gamma(X, X_test, no_of_comp, i) * K(z, X[i, :], 1) * X[i]

    for i in range(X.shape[0]):
        denominator += gamma(X, X_test, no_of_comp, i) * K(z, X[i, :], 1)

    return numerator / denominator


def p_of_z(X, X_test, no_of_comp):
    ret = 0.0
    for i in range(X.shape[0]):
        ret += gamma(X, X_test, no_of_comp, i) * K(X_test, X[i, :], 1)
    return ret  # TODO: add Omega value


def de_noise_image_multiple(X_train, X_test, num_components, num_iterations):
    iX = np.copy(X_train)

    results = np.zeros(X_test.shape)

    for index in range(X_test.shape[0]):
        print("****** starting on test image %s ******" % (index))
        current_z = None
        test_image = np.copy(X_test[index])
        for i in range(num_iterations):
            start_time = time()
            print("iteration %s" % (i+1))
            test_image = calculate_z(iX, test_image, num_components)
            pz = p_of_z(iX, test_image, num_components)
            if current_z is None:
                current_z = pz
            else:
                if current_z <= pz or pz is np.nan:
                    print("converged. Breaking.")
                    break
            print("p(z) = %s" % pz)
            # print("X_test at iteration %s: %s" % (i, X_test))
            print("iteration %s took %.01f" % (i+1, time() - start_time))
        results[index, :] = test_image
    return results



def de_noise_image(X_train, X_test, num_components, num_iterations):
    iX_test = np.copy(X_test)
    iX = np.copy(X_train)

    current_z = None

    for i in range(num_iterations):
        start_time = time()
        print("iteration %s" % (i+1))
        iX_test = calculate_z(iX, iX_test, num_components)
        pz = p_of_z(iX, iX_test, num_components)
        if current_z is None:
            current_z = pz
        else:
            if current_z <= pz:
                print("converged. Breaking.")
                break
        print("p(z) = %s" % pz)
        # print("X_test at iteration %s: %s" % (i, X_test))
        print("iteration %s took %.01f" % (i+1, time() - start_time))
    return iX_test

    # current_pz = None
    # de_noised_image = np.copy(noised_image)
    # for i in range(num_iterations):
    #     print("Starting iteration %s" % (i+1))
    #     de_noised_image = calculate_z(training_set, de_noised_image, num_components)
    #     pz = p_of_z(training_set, de_noised_image, num_components)
    #     if current_pz is None:
    #         current_pz = pz
    #     else:
    #         if current_pz <= pz:
    #             break
    # return de_noised_image


if __name__ == '__main__':
    train_count = 3000
    test_data_point = 777
    num_components = 1
    num_iterations = 1

    usps_data, usps_target = datasets.usps_scikit()
    X_train, X_test, Y_train, Y_test = train_test_split(usps_data, usps_target, train_size=0.7, random_state=42)
    X = X_train[0:train_count]
    clean_image = X_test[test_data_point]
    true_label = Y_test[test_data_point] - 1

    noisy_image = add_noise(clean_image, 'speckle')
    de_noised_image = de_noise_image(X, noisy_image, num_components, num_iterations)


    drawImage(clean_image, title="%s:%s recovered" % (true_label, num_components), dontDraw=False)
    drawImage(noisy_image, title="%s:%s recovered" % (true_label, num_components), dontDraw=False)
    drawImage(de_noised_image, title="%s:%s recovered" % (true_label, num_components), dontDraw=False)