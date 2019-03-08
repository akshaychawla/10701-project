import os
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import scipy.io as sio 
import h5py

def main():
    approach = 'ekm'
    # approach = 'llc'
    print('approach: ', approach)
    percentage_train = 0.8
    percentage_test = 0.2
    Cs= [0.1, 1.0, 10.0, 20.0, 100.0]

    # if approach == 'ekm':
    database_path = 'features/Caltech101/database.mat'
    # load ekm histogram
    database = h5py.File(database_path, 'r')
    database = database['database']
    numImages = int(database['imnum'][0][0])
    features = np.array(database['features_ekm'], dtype=float)
    features = features.T
    labels = np.array(database['label'][0], dtype=int)
    snumClasses = int(database['nclass'][0][0])
    # else:
    #     database_path = 'features_llc/Caltech101/database.mat'
    #     database = sio.loadmat(database_path)
    #     database = database['database']
    #     numImages = database['imnum'][0][0][0][0]
    #     features = database['features'][0][0]
    #     labels = database['label'][0][0]
    #     numClasses = database['nclass'][0][0][0][0]
    

    # setup data
    train_indices = np.array([], dtype=int)
    test_indices = np.array([], dtype=int)
    for cl in range(numClasses):
        indices = np.where(labels == cl + 1)[0]
        num = len(indices)
        indices = np.random.permutation(indices)
        train_indices = np.append(train_indices, indices[0:int(percentage_train*num)])
        test_indices = np.append(test_indices, indices[int(percentage_train*num)+1:-1])
    
    train_X = features[:, train_indices]
    train_labels = labels[train_indices]
    test_X = features[:,test_indices]
    test_labels = labels[test_indices]

    for c in Cs:
        print('c: ', c)
        svm_ln = svm.LinearSVC(C=c)
        # svm_ln = svm.SVR(kernel='linear', C=c)
        svm_ln.fit(train_X.T, train_labels)
        svm_train_results = svm_ln.predict(train_X.T)
        svm_test_results = svm_ln.predict(test_X.T)
        svm_test_error = float((svm_test_results != test_labels).mean())
        svm_train_error = float((svm_train_results != train_labels).mean())
        print('svm test error: ', svm_test_error)
        print('svm train error: ', svm_train_error)

    # softmax


if __name__ == "__main__":
    main()