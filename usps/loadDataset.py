import numpy as np 
import re
from scipy.io import loadmat
import os,sys

def read_mat_data():
	#http://www.cs.nyu.edu/~roweis/data.html
	path = os.path.realpath(__file__)
	file_path = os.path.abspath(os.path.join(path,'../data/usps_all.mat'))
	data = loadmat(file_path)
	return data['data'].T


def read_resampled_data():
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

if __name__ == '__main__':
	data_all = read_mat_data()
	data_test, data_train = read_resampled_data()
	