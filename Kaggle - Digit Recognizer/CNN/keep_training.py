from keras.models import load_model
import pandas as pd
import numpy as np
from printTrainingHistory import plotLossAndAcc
pd.options.mode.chained_assignment = None

PATH_TRAIN = "../data/train.csv"
#PATH_TRAIN = "./new_dataset.csv"

# loading training data
print(">>>loading data...")
labeled_images = pd.read_csv(PATH_TRAIN)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
print(">>>preprocessing data...")
images = images.astype('float32')
images /= 255
mean = np.mean(images)
images -= mean
images = np.asarray([ x.reshape(28,28,1) for x in images.as_matrix() ])
pm = []
for x  in labels.as_matrix():
    rl = [0,0,0,0,0,0,0,0,0,0]
    rl[x[0]] = 1
    pm.append(rl.copy())
labels = np.asarray(pm)

# setup an ANN
print('>>>loading nerual network...')
model = load_model('CNN4MNIST.h5fmodel')
model.summary()

# start training this model
print('>>>keep training model...')
history =  model.fit(images,labels,epochs=50,verbose=1,validation_split=0.2,batch_size=300)

#saving model
print('>>>done.\n>>>saving model...')
model.save('CNN4MNIST.h5fmodel')  # creates a HDF5 file 'my_model.h5'

# plot training history
plotLossAndAcc(history)
print(">>>all done.")