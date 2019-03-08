import numpy as np
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import multiprocessing
import skimage.io
import os
import pprint
from sklearn.model_selection import train_test_split
import sklearn

RESULTS_PATH = '../results/filter_responses'
BASE_PATH='../data/101_ObjectCategories'

if(__name__=='__main__'):

    # uncomment for visualization

    # paths = ['airplanes/image_0056.jpg', 'butterfly/image_0031.jpg', 'pizza/image_0020.jpg']
    # path_img = BASE_PATH+'/'+paths[0]
    # image = skimage.io.imread(path_img)
    # image = image.astype('float')/255
    # filter_responses = visual_words.extract_filter_responses(image)
    # util.display_filter_responses(filter_responses)
    # dictionary = np.load('../results/train_dictionary.npy')
	
    # for path in paths:
    #     image = skimage.io.imread(BASE_PATH+'/'+path)
    #     image = image.astype('float')/255
    #     wordmap = visual_words.get_visual_words(image,dictionary)
    #     #print(wordmap)
    #     util.save_wordmap(wordmap, os.path.basename(path))


    num_cores = multiprocessing.cpu_count()
    x, y = util.get_filenames(BASE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=10)
    print(len(X_train))
    alpha = 50 # number of pixels per image
    K = 100 # number of cluster centers
    visual_words.compute_dictionary(X_train, RESULTS_PATH,K, alpha, 8)
    files = os.listdir(RESULTS_PATH)	
    filter_responses = []
    for file in files:
        filter_responses.append(np.load(os.path.join(RESULTS_PATH,file)))	
    filter_responses = np.concatenate(filter_responses, axis = 0)
    K = 100
    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=-1).fit(filter_responses)
    dictionary = kmeans.cluster_centers_	
    np.save("../results/train_dictionary.npy", dictionary)
    print("dictionary done")
    visual_recog.build_recognition_system(X_train, y_train, K, num_workers=num_cores)
    np.set_printoptions(threshold=np.inf)
    conf, accuracy = visual_recog.evaluate_recognition_system(X_train, y_train, num_workers=num_cores)
    pprint.pprint(conf)
    print(accuracy)