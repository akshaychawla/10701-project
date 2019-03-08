import numpy as np
import multiprocessing
import queue
import imageio
import os,time
import math
import visual_words
import matplotlib.pyplot as plt
import skimage.io

def build_recognition_system(X, y, K, num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''
	# train_data = np.load("../data/train_data.npz")
	dictionary = np.load("../results/train_dictionary.npy")
	# labels = np.array(train_data['labels'].tolist())
	list_image_names = []
	SPM_layer_num = 3
	for path in X:
		list_image_names.append((path,dictionary, SPM_layer_num,K))
	pool = multiprocessing.Pool(num_workers)	
	results = pool.map(get_image_feature_wrapper, list_image_names)
	pool.close()
	pool.join()
	features = np.array(results)
	# print("features done\n",features.shape)
	np.savez("../results/trained_system.npz", dictionary=dictionary, features=features, labels=y, SPM_layer_num=SPM_layer_num)
	
def evaluate_recognition_system(X_test, y_test, num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (102,102)
	* accuracy: accuracy of the evaluated system
	'''
	# test_data = np.load("../data/test_data.npz")
	# test_labels = test_data['labels']

	trained_system = np.load("../results/trained_system.npz")
	dictionary = trained_system['dictionary']
	print(dictionary.shape)
	features = trained_system['features']
	train_labels = trained_system['labels']
	
	SPM_layer_num = trained_system['SPM_layer_num']	
	


	num_classes = len(set(y_test))
	conf_mtx = np.zeros((num_classes, num_classes))
	list_test_img = []
	for i, path in enumerate(X_test):
		list_test_img.append((i,path,dictionary,features, train_labels, y_test, SPM_layer_num))
	pool = multiprocessing.Pool(num_workers)
	results = pool.map(classify,list_test_img)
	pool.close()
	pool.join()
	# print("classification done")
	for train_label, test_label in results:
		conf_mtx[test_label][train_label] = conf_mtx[test_label][train_label] + 1
	accuracy = np.trace(conf_mtx)/ np.sum(conf_mtx)
	return conf_mtx, accuracy


def classify(args):
	i,path,dictionary,features, train_labels, test_labels, SPM_layer_num = args
	image = skimage.io.imread(path)
	image = image.astype('float')/255
	if(len(image.shape)!=3):
		image = visual_words.to_rgb(image)
	wordmap = visual_words.get_visual_words(image,dictionary)
	hist = get_feature_from_wordmap_SPM(wordmap,SPM_layer_num,dictionary.shape[0])	
	
	# top10_indices = np.argpartition(distance_to_set(hist, features),-10)[-10:]
	# (values,counts) = np.unique(top10_indices,return_counts=True)
	# train_index=np.argmax(counts)
	# train_index = np.argmax(distance_to_set(hist, features))

	train_index = np.argmax(distance_to_set(hist, features))	

	train_label = train_labels[train_index]
	test_label = test_labels[i]
	return train_label, test_label	

def get_image_feature_wrapper(args):
	file_path,dictionary,layer_num,K = args
	return get_image_feature(file_path,dictionary,layer_num,K)

def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''
	image = skimage.io.imread(file_path)
	image = image.astype('float')/255
	if(len(image.shape)!=3):
		image = visual_words.to_rgb(image)		
	wordmap = visual_words.get_visual_words(image,dictionary)
	spm_features = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
	return spm_features

def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
	'''
	This function returns the histogram
	intersection similarity between word hist 
	and each training sample as a vector of length T
	'''	
	# print(word_hist.shape, histograms.shape)
	return np.sum(np.minimum(word_hist, histograms), axis = 1)

def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	
	hist, _ = np.histogram(wordmap, bins = dict_size, density=True)	
	hist = np.nan_to_num(hist)	
	return hist

def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''	
	hist = []
	h = wordmap.shape[0]
	w = wordmap.shape[1]
	L = layer_num - 1
	for l in range(L,-1,-1):
		for i in range(0,2**l):
			for j in range(0,2**l):
				if(l==0 or l==1):
					weight = 2**(-l*1.0)
				else:
					weight = 2**(l*1.0-L-1)
				row_begin = math.floor((h/2**l)*i)
				row_end = math.floor((h/2**l)*(i+1))
				col_begin = math.floor((w/2**l)*j)
				col_end = math.floor((w/2**l)*(j+1))
				h_temp = (weight/(2**(2*l))) * get_feature_from_wordmap(wordmap[row_begin:row_end+1,col_begin:col_end+1], dict_size)
				hist.append(h_temp)
	ret = np.array(hist).flatten()
	return ret	