import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Data import datasets
from sklearn.decomposition import PCA, KernelPCA
from new_KernelPCA import de_noise_image

def drawImage(image, shape):
    plt.imshow(np.reshape(image, shape), cmap=cm.Greys_r)
    plt.axis('off')
    plt.show()

def saveImage(image, shape, name):
    plt.imshow(np.reshape(image, shape), cmap=cm.Greys_r)
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', pad_inches=0)

def pca_reduce(train_set, test_set, component_count, dev = None):
    pca = PCA(n_components=component_count)
    pca.fit(train_set)
    transformed_test = pca.transform(test_set)
    return pca.inverse_transform(transformed_test)


def kernel_pca_reduce(train_set, test_set, component_count, dev = None):
    if dev == None:
        kpca = KernelPCA(kernel="rbf", n_components=component_count, fit_inverse_transform=True)
    else:
        dim = np.shape(test_set)[1]
        kpca = KernelPCA(kernel="rbf", n_components=component_count, fit_inverse_transform=True, gamma = 1/(dim * 2 * dev**2))
    kpca.fit(train_set)
    transformed_test = kpca.transform(test_set)
    return kpca.inverse_transform(transformed_test)


def squared_distance(instance1, instance2):
    distance = 0
    for i in range(len(instance1)):
        distance += pow((instance1[i] - instance2[i]), 2)
    return distance


def toy1():
    linear_pca_score_matrix = generate_score_matrix(pca_reduce)
    kernel_pca_score_matrix = generate_score_matrix(de_noise_image)

    print(linear_pca_score_matrix)
    print(kernel_pca_score_matrix)
    print(linear_pca_score_matrix/kernel_pca_score_matrix)


def generate_score_matrix(pca_function, NOcenters = 11):
    deviations = [0.05, 0.1, 0.2, 0.4, 0.8]
    features = [i for i in range(1, 10)]
    score_matrix = np.zeros((len(deviations),len(features)))
    for dev_i, dev in enumerate(deviations):
        train, test, centers = datasets.toy1(dev, NOcenters)
        for f_i, feature_count in enumerate(features):
            denoised_test = pca_function(train, test, feature_count, 5) # Optional: include parameter 'dev'. Only affects KPCA. Seems to yield worse results.
            score_matrix[dev_i][f_i] = calculate_score(denoised_test, centers)
    return score_matrix


def calculate_score(data, centers):
    return np.mean([squared_distance(datapoint, center) for datapoint, center in zip(data, centers)])


def add_2D_noise(list, noise_area = [0, 1, 0, 1], noise_amount = 100):
    noisy_image = list(image)
    for i in range(noise_amount):
        noisy_image.append([np.random.uniform(noise_area[0], noise_area[1]), np.random.uniform(noise_area[2], noise_area[3])])
    return noisy_image


def toy2():
    toy2circle(kernel_pca_reduce, 4)
    #toy2square(kernel_pca_reduce, 4)


def toy2circle(pca_function, features):
    axis = [-1.2, 1.2, -.2, 1.2]
    halfcircle = datasets.half_circle()
    noisy_halfcircle = add_2D_noise(halfcircle, axis)

    denoised_halfcircle = pca_function(halfcircle, noisy_halfcircle, features)

    halfcircle = np.array(halfcircle)
    noisy_halfcircle = np.array(noisy_halfcircle)

    plottoy2figure(halfcircle, noisy_halfcircle, denoised_halfcircle, axis)


def toy2square(pca_function, features):
    axis = [-.3, 1.3, -.3, 1.3]
    square = datasets.square()
    noisy_square = add_2D_noise(square, axis)

    denoised_square = pca_function(square, noisy_square, features)

    square = np.array(square)
    noisy_square = np.array(noisy_square)

    plottoy2figure(square, noisy_square, denoised_square, axis)


def plottoy2figure(original, noisy, denoised, axis = [0, 1, 0, 1]):
    plt.figure(1)
    plt.axis(axis)
    plt.plot(noisy[:,0], noisy[:,1], 'o')

    plt.figure(2)
    plt.axis(axis)
    plt.plot(original[:,0], original[:,1], 'ok')
    plt.plot(denoised[:,0], denoised[:,1], 'or')
    plt.show()


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

def generate_figure4():
    usps_data = datasets.usps_stored()

    print("Starting to generate number images")
    for nr in range(0,10):
        print("Generating images for nr {0}".format(nr))
        #drawImage(usps_data["test_clean"][nr],(16,16))
        saveImage(usps_data["test_clean"][nr], (16,16), "data/denoised_usps/{0}/original_clean.png".format(nr))

        #drawImage(usps_data["test_speckle"][nr],(16,16))
        saveImage(usps_data["test_speckle"][nr], (16,16), "data/denoised_usps/{0}/original_speckle.png".format(nr))

        #drawImage(usps_data["test_gaussian"][nr],(16,16))
        saveImage(usps_data["test_gaussian"][nr], (16,16), "data/denoised_usps/{0}/original_gaussian.png".format(nr))
        for comp_i, comp in enumerate([1,4,16,64,256]):
            ###############PCA###############
            print("Starting {0} training with {1} noise and component_n {2}".format("linearPCA","speckle",comp))
            linear_speckle_numbers = pca_reduce(usps_data["train"], usps_data["test_speckle"], comp)
            #drawImage(linear_speckle_numbers[nr],(16,16))
            saveImage(linear_speckle_numbers[nr], (16,16), "data/denoised_usps/{0}/{2}_{3}_{1}.png".format(nr, comp,"linearPCA","speckle"))

            print("Starting {0} training with {1} noise and component_n {2}".format("linearPCA","gaussian",comp))
            linear_gaussian_numbers = pca_reduce(usps_data["train"], usps_data["test_gaussian"],comp)
            #drawImage(linear_gaussian_numbers[nr],(16,16))
            saveImage(linear_gaussian_numbers[nr], (16,16), "data/denoised_usps/{0}/{2}_{3}_{1}.png".format(nr, comp,"linearPCA","gaussian"))

            ###############Kernel PCA###############
            print("Starting {0} training with {1} noise and component_n {2}".format("kernelPCA","speckle",comp))
            kernel_speckle_numbers = kernel_pca_reduce(usps_data["train"], usps_data["test_speckle"], comp)
            #drawImage(kernel_speckle_numbers[nr],(16,16))
            saveImage(kernel_speckle_numbers[nr], (16,16), "data/denoised_usps/{0}/{2}_{3}_{1}.png".format(nr, comp,"kernelPCA","speckle"))

            print("Starting {0} training with {1} noise and component_n {2}".format("linear","gaussian",comp))
            kernel_gaussian_numbers = kernel_pca_reduce(usps_data["train"], usps_data["test_gaussian"],comp)
            #drawImage(kernel_gaussian_numbers[nr],(16,16))
            saveImage(kernel_gaussian_numbers[nr], (16,16), "data/denoised_usps/{0}/{2}_{3}_{1}.png".format(nr, comp,"kernelPCA","gaussian"))

def generate_figure3():
    usps_data = datasets.usps_stored()
    scores = []
    saveImage(usps_data["test_clean"][2],(16,16),"data/fraction_usps/three_clean.png")
    for comp_count in range(1,21):
        print("Starting training for {0} components for number 3".format(comp_count))
        pca_denoised = pca_reduce(usps_data["train"], usps_data["test_gaussian"],comp_count)
        pca_score = np.sum((pca_denoised[2]-usps_data["test_clean"][2])**2)

        kernel_pca_denoised = kernel_pca_reduce(usps_data["train"], usps_data["test_gaussian"],comp_count)
        kernel_pca_score = np.sum((kernel_pca_denoised[2]-usps_data["test_clean"][2])**2)
        final_score = pca_score/kernel_pca_score
        scores.append(round(final_score,2))
        saveImage(pca_denoised[2],(16,16),"data/fraction_usps/three_{0}_{1}.png".format('linearPCA',comp_count))
        saveImage(kernel_pca_denoised[2],(16,16),"data/fraction_usps/three_{0}_{1}.png".format('kernelPCA',comp_count))

    f = open("data/fraction_usps/scores.txt","w")
    f.write("&  \\tiny{"+"}\n &  \\tiny{".join(map(str, scores))+"}")



if __name__ == '__main__':
    """
    TRAINING_COUNT = 300
    usps_all = datasets.usps()
    usps_train, noisy_usps_test = datasets.usps(noise_type='speckle')
    denoised_test = reduce_noise(usps_train, noisy_usps_test)
    drawImage(noisy_usps_test[3][0],(16,16))
    drawImage(denoised_test[3][0],(16,16))
    """

    generate_figure3()
    #toy1()
    #generate_figure4()
    quit()

    #test_image_size = 16
    #image_circle = datasets.half_circle(test_image_size)
    #image_box = datasets.box(test_image_size)
    #drawImage(image_box,(test_image_size,test_image_size))
    #drawImage(image_circle,(test_image_size,test_image_size))

    #drawImage(noisy_image,(16,16))
    #drawImage(usps_test[8][0],(16,16))
    #drawImage(mnist_train[8][0],(28,28))
    # drawImage(usps_test[8][0],(16,16))
