from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from helper import saveResult
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.ndimage import interpolation
from multiprocessing import Pool
from sklearn import preprocessing

# this puts input as list of 28x28 matrices, upright, with normalized greyscale values 
x_train_in = np.array(pd.read_csv('MNIST_Xtrain.csv', header=None)).astype('float')
x_train_img = [np.transpose(x.reshape(28,28)) for x in x_train_in]

x_test_in = np.array(pd.read_csv('MNIST_Xtestp.csv', header=None)).astype('float')
x_test_img = [np.transpose(x.reshape(28,28)) for x in x_test_in]

y_train = np.array(pd.read_csv('MNIST_ytrain.csv', header=None)).astype('int').reshape(x_train_in.shape[0])

# Deskew preprocessing, credits: https://fsix.github.io/mnist/Deskewing.html
def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)

def deskewSet(aSet):
    workers = Pool()
    deskewed = workers.map(deskew, aSet)
    workers.close()
    workers.join()
    return deskewed

# Deskew training set, then append deskewed images to training set
# Note x_test is NOT deskewed
x_train_deskewed_img = deskewSet(x_train_img)
x_train_img = np.append(x_train_img, x_train_deskewed_img, axis=0)
x_test_deskewed_img = deskewSet(x_test_img)
x_test_img = x_test_deskewed_img
y_train = np.append(y_train, y_train, axis=0)

# Finally, ravel x_train_img so it is can be used for training
x_train = [x.transpose().ravel() for x in x_train_img]
x_test = [x.transpose().ravel() for x in x_test_img]

# Scale samples, might be useful, must be done after ravel
# This is done to both x_train and x_test
preprocessing.scale(x_train, axis=1, copy=False)
preprocessing.scale(x_test, axis=1, copy=False)

# PCA
# 40 and 50 both seem OK, beyond 50 accuracy begins to drop
# 60 works well with deskewing
pca = PCA(n_components=50)
pca.fit(x_train)
x_train_reduce = pca.transform(x_train)
x_test_reduce = pca.transform(x_test)

clf = KNeighborsClassifier(n_neighbors=1, weights='distance', p=2, n_jobs=-1)
clf.fit(x_train_reduce, y_train)

pred = clf.predict(x_test_reduce)

saveResult('20667932.csv', pred)
