import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

PATH_TRAIN = "../data/train.csv"

# loading training data
print(">>>loading data...")
labeled_images = pd.read_csv(PATH_TRAIN)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
print(">>>preprocessing data...")
images[images>0]=1
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.798,test_size=0.202, random_state=0)

# start training this model
print('>>>training model...')
KNN =  neighbors.KNeighborsClassifier(n_neighbors=10,n_jobs=4)
KNN.fit(train_images,train_labels.values.ravel() )
# score the KNN classifier
print(">>>scoring...")
s = KNN.score(test_images,test_labels.values.ravel() )
print(">>>Mean accuracy is:",s)

# save model
print('>>>done.\n>>>saving model...')
data = KNN.get_params()
with open("KNN.model","w") as f:
    f.write(str(data))
print(">>>all done.")

############################## for predict
PATH_TEST = "../data/test.csv"

# load test data
print('>>>loading test data...')
test_data=pd.read_csv(PATH_TEST)
test_data[test_data>0]=1

# eval test data
print('>>>evaluating test data...')
result = KNN.predict(test_data)

# output result
print(">>>saving results...")
df = pd.DataFrame({'Label':result})
df.index += 1
df.index.name='ImageId'
df.to_csv('results.csv')
print(">>>done.")