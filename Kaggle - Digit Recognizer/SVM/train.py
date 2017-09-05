import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.externals import joblib
import disp_multiple_images

PATH_TRAIN = "../data/train.csv"

# load training data
print(">>>loading data...")
labeled_images = pd.read_csv(PATH_TRAIN)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.9, random_state=0)
del labeled_images

# make picture from grayscale to turely black white image
print(">>>transforming data...")
test_images[test_images>0]=1        # set all pixels that noezero to 1
train_images[train_images>0]=1      # zero stays the same.

# draw picture
if False:
    img=train_images.as_matrix()
    print("before reshape all image to 28x28:",img.shape)
    print("one of the image size is:",len(img[0]))
    img =  np.asarray([ i.reshape(28,28) for i in img])
    print("after reshape all image to 28x28:",img.shape)
    disp_multiple_images.show_images(img[:16],4)

# train with SVM.SVC()
print(">>>start training model... with svm.SVC()")
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print(">>>Model training has been finished.Running test for current model...")
print(">>>Test Score:",clf.score(test_images,test_labels),"\n>>>saving model...")
joblib.dump(clf, 'svm_svc.model') 
print(">>>model saved.\n>>>done.")
