import numpy as np
from scipy.io import loadmat
from skimage.draw import circle_perimeter
from skimage.util import random_noise
import os
import pickle
import math
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

class datasets:

	@staticmethod
	def usps_scikit():
		usps = fetch_mldata('usps', data_home="mldata")
		return usps.data, usps.target

	@staticmethod
	def usps_stored():
		data = {}
		path = os.path.realpath(__file__)
		train_path = os.path.abspath(os.path.join(path,'../data/train_data.mat'))
		data["train"] = loadmat(train_path)['X']
		test_path = os.path.abspath(os.path.join(path,'../data/test.mat'))
		test = loadmat(test_path)
		data["test_speckle"] = test['speckle']
		data["test_gaussian"] = test['gaussian']
		data["test_clean"] = test['test']
		return data

	@staticmethod
	def usps_stored_test():
		data = {}
		path = os.path.realpath(__file__)
		train_path = os.path.abspath(os.path.join(path,'../data/test.mat'))
		test = loadmat(train_path)
		speckle = test['speckle']
		return speckle

	@staticmethod
	def usps(noise_type=None):
		usps = fetch_mldata('usps')
		train_dataset = [[] for _ in range(10)]
		test_dataset = [[] for _ in range(10)]
		X_train, X_test, Y_train, Y_test = train_test_split(usps.data, usps.target, train_size=0.7, random_state=42)
		for i, nr in enumerate(Y_train):
			train_dataset[nr-1].extend([X_train[i]])

		for i, nr in enumerate(Y_test):
			test_dataset[nr-1].extend([X_test[i]])

		train_dataset = np.array(train_dataset)
		test_dataset = np.array(test_dataset)
		if noise_type:
			return train_dataset, datasets.add_noise(test_dataset,noise_type)
		return train_dataset, test_dataset

	@staticmethod
	def mnist(noise_type=None):
		#https://code.google.com/p/kernelmachine/
		path = os.path.realpath(__file__)
		file_path = os.path.abspath(os.path.join(path,'../data/mnist_all.mat'))
		data = loadmat(file_path)
		d_train = [[] for _ in range(10)]
		d_test = [[] for _ in range(10)]
		for i in range(10):
			d_test[i].extend(np.true_divide(data['test{0}'.format(i)],256))
			d_train[i].extend(np.true_divide(data['train{0}'.format(i)],256))
		if noise_type:
			return np.array(d_train), np.array(datasets.add_noise(d_test,noise_type))
		else:
			return np.array(d_train), np.array(d_test)

	@staticmethod
	def usps_resampled(noise_type=None):
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
		if noise_type:
			return np.array(trainingset), np.array(datasets.add_noise(testset,noise_type))
		else:
			return np.array(trainingset), np.array(testset)

	@staticmethod
	def add_noise(dataset, noise_type):
		for i,nr in enumerate(dataset):
			for j,image in enumerate(nr):
				if noise_type == 'speckle':
					dataset[i][j] = random_noise(dataset[i][j], mode='s&p', salt_vs_pepper=0.5, amount=0.4)
				elif noise_type == 'gaussian':
					dataset[i][j] = random_noise(dataset[i][j], mode='gaussian', var=0.5)
		return dataset

	@staticmethod
	def toy1(standard_dev, dimensions = 10, NOcenters = 11):
		#return pickle.load( open("data/toy_data({0}).p".format(standard_dev), "rb"))
		
		train = []
		test = []
		cov = np.eye(dimensions) * standard_dev**2
		centers = []
		for i in range(NOcenters):
			center = np.random.uniform(-1.0, 1.0, dimensions)
			centers.extend([center for _ in range(33)])
			train.extend(np.random.multivariate_normal(center, cov, 60))
			test.extend(np.random.multivariate_normal(center, cov, 5))
		#pickle.dump((train,test,centers), open("data/toy_data({0}).p".format(standard_dev), "wb"))
		return np.array(train), np.array(test), np.array(centers)		
		

	@staticmethod
	def square():
		square = []
		for i in range(100):
			square.append([0, i/100])
		for i in range(100):
			square.append([i/100, 1])
		for i in range(100):
			square.append([1, 1 - i/100])
		for i in range(101):
			square.append([1 - i/100, 0])
		return square


	@staticmethod
	def half_circle():
		circle_center = [0, 0]
		circle_radius = 1
		circle = []
		for i in range(200):
			theta = i/200 * math.pi
			circle.append([circle_center[0] + circle_radius * math.cos(theta), circle_center[1] + circle_radius * math.sin(theta)])
		return circle


train, test = datasets.mnist()



