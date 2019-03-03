import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import skimage.io
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random
import math
import shutil
from sklearn.model_selection import train_test_split

RESULTS_PATH = '../results/filter_responses'
BASE_PATH='../data/101_ObjectCategories'

def to_rgb(img):
	if(img.ndim == 2):
		print("GRAYSCALE")
		rgb_img = np.empty((img.shape[0], img.shape[1]))
		rgb_img[:,:,0] = img
		rgb_img[:,:,1] = img
		rgb_img[:,:,2] = img
		return rgb_img
	elif(img.ndim == 3 and img.shape[2]>3):
		print("4D", img.shape[2])
		return img[...,0:3]

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	if(image.shape[2]!=3): #if not RGB, replicate across 3 axes
		image = to_rgb(image)

	scales = [1, 2, 4, 8, math.sqrt(2)*8]
	image = skimage.color.rgb2lab(image)	
	filter_responses=[]
	for scale in scales:
		gaussian_filter_response = list()
		gaussian_laplace_response = list()
		gaussian_x_response = list()
		gaussian_y_response = list()
		for num_channels in range(image.shape[2]):
			gaussian_filter_response.append(scipy.ndimage.gaussian_filter(image[:,:,num_channels], scale))
			gaussian_laplace_response.append(scipy.ndimage.gaussian_laplace(image[:,:,num_channels], scale))
			gaussian_x_response.append(scipy.ndimage.gaussian_filter(image[:,:,num_channels], scale, order = [0, 1]))
			gaussian_y_response.append(scipy.ndimage.gaussian_filter(image[:,:,num_channels], scale, order = [1, 0]))
		filter_responses.append(np.dstack(tuple(gaussian_filter_response)))
		filter_responses.append(np.dstack(tuple(gaussian_laplace_response)))
		filter_responses.append(np.dstack(tuple(gaussian_x_response)))
		filter_responses.append(np.dstack(tuple(gaussian_y_response)))
	stacked_filter_responses = np.dstack(tuple(filter_responses))	
	return stacked_filter_responses

def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''	
	h,w = image.shape[0], image.shape[1]
	wordmap = np.zeros((h,w))
	filter_response = extract_filter_responses(image)
	filter_response = np.reshape(filter_response, (-1, filter_response.shape[2]))
	distance_matrix = scipy.spatial.distance.cdist(filter_response, dictionary)	
	wordmap = np.argmin(distance_matrix, axis=1)	
	wordmap = np.reshape(wordmap, (h,w))
	return wordmap
	

def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	
	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''
	i,alpha,image_path = args
	image = skimage.io.imread(image_path)
	image = image.astype('float')/255
	if(image.shape[2]!=3):
		image = to_rgb(image)
	filter_responses = extract_filter_responses(image)
	shuffled_filter_responses = np.random.permutation(filter_responses)
	np.save(RESULTS_PATH+"/image_"+str(i), shuffled_filter_responses[0:alpha,0,:])

def compute_dictionary(list_image_names, results_path, K, alpha, num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
    * X: image paths
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''
    
    # train_data = np.load("../data/train_data.npz")
	# alpha = 500
	# list_image_names = []
	# for path in train_data['image_names'].tolist():
	# 	list_image_names.append(path[0])
	args = [(i, alpha, image_names) for i, image_names in enumerate(list_image_names)]
	if(os.path.exists(results_path)):
		shutil.rmtree(results_path)
	os.makedirs(results_path)
	pool = multiprocessing.Pool(num_workers)
	pool.map(compute_dictionary_one_image, args)
	pool.close()
	pool.join()
	files = os.listdir(results_path)	
	filter_responses = []
	for file in files:
		filter_responses.append(np.load(os.path.join(results_path,file)))	
	filter_responses = np.concatenate(filter_responses, axis = 0)
	# K = 300
	kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=-1).fit(filter_responses)
	dictionary = kmeans.cluster_centers_	
	np.save("train_dictionary.npy", dictionary)

if(__name__=='__main__'):
    x, y = util.get_filenames(BASE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=10)
    alpha = 500
    K = 300
    compute_dictionary(X_train, RESULTS_PATH,K, alpha, 8)