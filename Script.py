import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Data import datasets


def drawImage(image):
    plt.imshow(np.reshape(image,(16,16)), cmap = cm.Greys)
    plt.show()

if __name__ == '__main__':
    usps_all = datasets.usps()
    usps_test, usps_train = datasets.usps_resampled()
    toy1_train,toy1_test = datasets.toy1(0.05)

    drawImage(usps_test[8][0])
    print(toy1_test.shape, toy1_train.shape)