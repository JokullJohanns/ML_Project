import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Data import datasets


def drawImage(image,shape):
    plt.imshow(np.reshape(image,shape), cmap = cm.Greys)
    plt.show()

if __name__ == '__main__':
	usps_all = datasets.usps()
	usps_train, usps_test = datasets.usps_resampled()
	toy1_train, toy1_test = datasets.toy1(0.05)
	mnist_train, mnist_test = datasets.mnist()
	drawImage(usps_test[8][0],(16,16))
	drawImage(mnist_train[8][0],(28,28))