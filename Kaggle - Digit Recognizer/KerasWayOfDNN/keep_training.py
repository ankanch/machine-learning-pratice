from keras.models import load_model
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

PATH_TRAIN = "../data/train.csv"

# loading training data
print(">>>loading data...")
labeled_images = pd.read_csv(PATH_TRAIN)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
print(">>>preprocessing data...")
images[images>0]=1
images = images.as_matrix().reshape(images.shape[0], 28, 28,1)
pm = []
for x  in labels.as_matrix():
    rl = [0,0,0,0,0,0,0,0,0,0]
    rl[x[0]] = 1
    pm.append(rl.copy())
    #print(x[0],":",rl)
labels = np.asarray(pm)
print(">>>images.shape=",images.shape,"\tlabels.shape=",labels.shape)

# setup an ANN
print('>>>loading nerual network...')
model = load_model('digital_recog_w_sequentialDenseNN.h5fmodel')

# start training this model
print('>>>keep training model...')
model.fit(images,labels,epochs=10,verbose=1,validation_split=0.2)

#saving model
print('>>>done.\n>>>saving model...')
model.save('digital_recog_w_sequentialDenseNN.h5fmodel')  # creates a HDF5 file 'my_model.h5'
print(">>>all done.")