import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.optimizers import Adam

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
#images[images>0]=1
images = images.astype('float32')
images /= 255
mean = np.mean(images)
images -= mean
images = images.as_matrix()
pm = []
for x  in labels.as_matrix():
    rl = [0,0,0,0,0,0,0,0,0,0]
    rl[x[0]] = 1
    pm.append(rl.copy())
labels = np.asarray(pm)
print(">>>images.shape=",images.shape,"\tlabels.shape=",labels.shape)
# setup an ANN
print('>>>setup sequential nerual network...')
model = Sequential()
model.add(Dense(units=512,activation='relu',input_shape=(784,) ))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10,activation='softmax'))
model.summary()
print('>>>compiling...')
model.compile(optimizer=Adam(lr=0.001),loss='mean_squared_error',metrics=['accuracy'])

# start training this model
print('>>>training model...')
history = model.fit(images,labels,epochs=130,verbose=1,validation_split=0.2,batch_size=300)

# save model
print('>>>done.\n>>>saving model...')
model.save('digital_recog_w_sequentialDenseNN.h5fmodel')  # creates a HDF5 file 'my_model.h5'

# plot training history
plotLossAndAcc(history)
print(">>>all done.")


