import numpy as np
from scipy.io import loadmat
import os


class datasets:

	@staticmethod
	def usps():
		#http://www.cs.nyu.edu/~roweis/data.html
		path = os.path.realpath(__file__)
		file_path = os.path.abspath(os.path.join(path,'../data/usps_all.mat'))
		data = loadmat(file_path)
		return data['data'].T

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
		return np.array(testset), np.array(trainingset)

	@staticmethod
	def toy1(variance):
		d_train = []
		d_test = []
		mean = np.array([0 for _ in range(10)])
		cov = np.eye(10) * variance
		for i in range(11):
			test_mean = np.random.uniform(-1.0, 1.0, 10)
			d_train.extend(np.random.multivariate_normal(mean, cov, 100))
			d_test.extend(np.random.multivariate_normal(test_mean, cov, 33))
		return np.array(d_train), np.array(d_test)


if __name__ == '__main__':
	usps_all = datasets.usps()
	usps_test, usps_train = datasets.usps_resampled()
	toy1_train,toy1_test = datasets.toy1(0.05)
	