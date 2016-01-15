import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Data import datasets

from sklearn.decomposition import PCA, KernelPCA

def drawImage(image, shape):
    plt.imshow(np.reshape(image, shape), cmap=cm.Greys)
    plt.show()

def pca_reduce(train_set, test_set, component_count):
    pca = PCA(n_components=component_count)
    pca.fit(train_set)
    transformed_test = pca.transform(test_set)
    return pca.inverse_transform(transformed_test)

def kernel_pca_reduce(train_set, test_set, component_count):
    kpca = KernelPCA(kernel="rbf", n_components=component_count, fit_inverse_transform=True)
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

def reduce_noise(train, test, components_n=256):
    training_set = []
    test_set = []

    #join the data into one array for training
    for i in range(0, 10):
        training_set.extend(train[i][0:TRAINING_COUNT])
        test_set.extend(test[i][0:TRAINING_COUNT])
    denoised = kernel_pca_reduce(training_set, test_set, components_n)
    test_denoised = []

    #reshape the array back to normal
    for i in range(0,TRAINING_COUNT*10,TRAINING_COUNT):
        test_denoised.append(denoised[i:i+TRAINING_COUNT])
    return test_denoised


if __name__ == '__main__':
    TRAINING_COUNT = 300
    usps_all = datasets.usps()
    usps_train, noisy_usps_test = datasets.usps(noise_type='speckle')
    denoised_test = reduce_noise(usps_train, noisy_usps_test)
    drawImage(noisy_usps_test[3][0],(16,16))
    drawImage(denoised_test[3][0],(16,16))

    quit()
    linear_pca_score_matrix = generate_toy_example(pca_reduce)
    kernel_pca_score_matrix = generate_toy_example(kernel_pca_reduce)

    print(kernel_pca_score_matrix/linear_pca_score_matrix)
    print(linear_pca_score_matrix/kernel_pca_score_matrix)

    #test_image_size = 16
    #image_circle = datasets.half_circle(test_image_size)
    #image_box = datasets.box(test_image_size)
    #drawImage(image_box,(test_image_size,test_image_size))
    #drawImage(image_circle,(test_image_size,test_image_size))

    #drawImage(noisy_image,(16,16))
    #drawImage(usps_test[8][0],(16,16))
    #drawImage(mnist_train[8][0],(28,28))
    # drawImage(usps_test[8][0],(16,16))
