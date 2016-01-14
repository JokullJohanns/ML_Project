import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Data import datasets
from skimage.util import random_noise
from sklearn.decomposition import PCA, KernelPCA

def drawImage(image, shape):
    plt.imshow(np.reshape(image, shape), cmap=cm.Greys)
    plt.show()


def add_noise(image, noise_type):
    if noise_type == 'speckle':
        image = random_noise(image, mode='s&p', salt_vs_pepper=0.5, amount=0.4)
    elif noise_type == 'gaussian':
        image = random_noise(image, mode='gaussian', var=0.5)
    return image

def pca_reduce(train_set, test_set, component_count):
    pca = PCA(n_components=component_count)
    pca.fit(train_set)
    transformed_test = pca.transform(test_set)
    return pca.inverse_transform(transformed_test)

def kernel_pca_reduce(train_set, test_set, component_count):
    kpca = KernelPCA(kernel="rbf",n_components=component_count, fit_inverse_transform=True, gamma=1)
    kpca.fit(train_set)
    transformed_test = kpca.transform(test_set)
    return kpca.inverse_transform(transformed_test)

def squared_distance(instance1, instance2):
    distance = 0
    for i in range(len(instance1)):
        distance += pow((instance1[i] - instance2[i]), 2)
    return distance

def calculate_score(data, centers):
    return np.mean([squared_distance(datapoint, center) for datapoint, center in zip(data, centers)])

def generate_toy_example(pca_function):
    deviations = [0.05, 0.1, 0.2, 0.4, 0.8]
    features = [i for i in range(1,10)]
    score_matrix = np.zeros((len(deviations),len(features)))
    for dev_i, dev in enumerate(deviations):
        toy1_train, toy1_test, toy1_test_means = datasets.toy1(dev)
        for f_i, feature_count in enumerate(features):
            denoised_features = pca_function(toy1_train, toy1_test, feature_count)
            score_matrix[dev_i][f_i] = calculate_score(denoised_features, toy1_test_means)
    return score_matrix


if __name__ == '__main__':
    usps_all = datasets.usps()
    usps_train, usps_test = datasets.usps_resampled()
    toy1_train, toy1_test, toy1_test_means = datasets.toy1(0.05)
    mnist_train, mnist_test = datasets.mnist()


    linear_pca_score_matrix = generate_toy_example(pca_reduce)
    kernel_pca_score_matrix = generate_toy_example(kernel_pca_reduce)

    print(kernel_pca_score_matrix/linear_pca_score_matrix)
    print(linear_pca_score_matrix/kernel_pca_score_matrix)

    test_image_size = 16
    image_circle = datasets.half_circle(test_image_size)
    image_box = datasets.box(test_image_size)
    #drawImage(image_box,(test_image_size,test_image_size))
    #drawImage(image_circle,(test_image_size,test_image_size))

    noisy_image=add_noise(usps_test[8][0], 'gaussian')
    #drawImage(noisy_image,(16,16))
    #drawImage(usps_test[8][0],(16,16))
    #drawImage(mnist_train[8][0],(28,28))
    # drawImage(usps_test[8][0],(16,16))
