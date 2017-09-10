import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam,Adadelta,RMSprop

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

# setup an CNN
print(">>>setup model...")
model = Sequential()
model.add(Conv2D(32, kernel_size=(4,4),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss="mean_squared_error",optimizer=RMSprop(),metrics=['accuracy'])

# train
print(">>>training model...")
history = model.fit(images,labels,batch_size=300,epochs=400,verbose=1,validation_split=0.1)

# save model
print('>>>done.\n>>>saving model...')
model.save('CNN4MNIST.h5fmodel')  # creates a HDF5 file 'my_model.h5'

# plot training history
plotLossAndAcc(history)
print(">>>all done.")