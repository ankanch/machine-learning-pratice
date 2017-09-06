import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense,Flatten

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
    print(x[0],":",rl)
labels = np.asarray(pm)
print(">>>images.shape=",images.shape,"\tlabels.shape=",labels.shape)
# setup an ANN
print('>>>setup sequential nerual network...')
model = Sequential()
model.add(Dense(units=784,activation='sigmoid',input_shape=(28,28,1) ))
model.add(Flatten())
model.add(Dense(units=10,activation='sigmoid'))
print('>>>model.input_shape=',model.input_shape,"\tmodel.output_shape=",model.output_shape)
print('>>>compiling...')
model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])

# start training this model
print('>>>training model...')
model.fit(images,labels,epochs=2,verbose=1,validation_split=0.2)

# save model
print('>>>done.\n>>>saving model...')
model.save('digital_recog_w_sequentialDenseNN.h5fmodel')  # creates a HDF5 file 'my_model.h5'
print(">>>all done.")

