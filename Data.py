import numpy as np
from scipy.io import loadmat
from skimage.draw import circle_perimeter
import os
import pickle


class datasets:

	@staticmethod
	def usps():
		#http://www.cs.nyu.edu/~roweis/data.html
		path = os.path.realpath(__file__)
		file_path = os.path.abspath(os.path.join(path,'../data/usps_all.mat'))
		data = loadmat(file_path)
		return data['data'].T

	@staticmethod
	def mnist():
		#https://code.google.com/p/kernelmachine/
		path = os.path.realpath(__file__)
		file_path = os.path.abspath(os.path.join(path,'../data/mnist_all.mat'))
		data = loadmat(file_path)
		d_train = [[] for _ in range(10)]
		d_test = [[] for _ in range(10)]
		for i in range(10):
			d_test[i].extend(np.true_divide(data['test{0}'.format(i)],256))
			d_train[i].extend(np.true_divide(data['train{0}'.format(i)],256))
		return d_train, d_test

	@staticmethod
	def usps_resampled():
		#Probably won't use this. Will keep this here until we figure out what to do with this
		#http://www.gaussianprocess.org/gpml/data/
		path = os.path.realpath(__file__)
		file_path = os.path.abspath(os.path.join(path,'../data/usps_resampled.mat'))
		data = loadmat(file_path)
		x = data['train_patterns'].T   # train patterns
		y = data['train_labels'].T     # train_labels
		xx = data['test_patterns'].T   # test patterns
		yy = data['test_labels'].T     # test labels

		testset = [[] for _ in range(10)]
		trainingset = [[] for _ in range(10)]

		for i, (test_labels, train_labels) in enumerate(zip(y,yy)):
			testLabel = np.where(test_labels == 1)[0][0]
			trainLabel = np.where(train_labels == 1)[0][0]
			testset[testLabel].append(x[i])
			trainingset[trainLabel].append(xx[i])
		return np.array(trainingset), np.array(testset)

	@staticmethod
	def toy1(standard_dev):
		return pickle.load( open("data/toy_data({0}).p".format(standard_dev), "rb"))
		'''
		d_train = []
		d_test = []
		mean = np.array([0 for _ in range(10)])
		cov = np.eye(10) * standard_dev**2
		test_means = []
		for i in range(11):
			test_mean = np.random.uniform(-1.0, 1.0, 10)
			test_means.extend([test_mean for _ in range(33)])
			d_train.extend(np.random.multivariate_normal(mean, cov, 100))
			d_test.extend(np.random.multivariate_normal(test_mean, cov, 33))
		pickle.dump((d_train,d_test,test_means), open("data/toy_data({0}).p".format(standard_dev), "wb"))
		return np.array(d_train), np.array(d_test), np.array(test_means)
		'''



	@staticmethod
	def box(image_size):
		image = np.zeros((image_size, image_size))
		boxRange = range(int(image_size*0.2),image_size-int(image_size*0.2))
		for i in boxRange:
			image[min(boxRange)][i] = 1
			image[max(boxRange)][i] = 1
			image[i][min(boxRange)] = 1
			image[i][max(boxRange)] = 1
		return image.flatten()

	@staticmethod
	def half_circle(image_size):
		circle_center = int(image_size/2)
		circle_radius = int((circle_center/2)*1.4)
		img = np.zeros((image_size, image_size), dtype=np.uint8)
		rr, cc = circle_perimeter(circle_center, circle_center, circle_radius)
		img[rr, cc] = 1
		for i in range(circle_center, image_size):
			for j in range(0,image_size):
				img[i][j] = 0
		return img


train, test = datasets.mnist()



