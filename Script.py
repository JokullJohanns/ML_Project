import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Data import datasets
from skimage.util import random_noise

def drawImage(image,shape):
        plt.imshow(np.reshape(image,shape), cmap = cm.Greys)
        plt.show()

def add_noise(image, noise_type):
        if noise_type=='speckle':
                image= random_noise(image, mode='s&p', salt_vs_pepper=0.5, amount=0.4)
        elif noise_type=='gaussian':
                image= random_noise(image, mode='gaussian', var=0.5)
        return image

if __name__ == '__main__':
        usps_all = datasets.usps()
        usps_train, usps_test = datasets.usps_resampled()
        toy1_train, toy1_test = datasets.toy1(0.05)
        mnist_train, mnist_test = datasets.mnist()
        noisy_image=add_noise(usps_test[8][0], 'speckle')
        drawImage(noisy_image,(16,16))
        #drawImage(mnist_train[8][0],(28,28))
