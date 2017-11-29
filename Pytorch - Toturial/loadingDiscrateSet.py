import os
import numpy as np
from matplotlib import pyplot as plt

PATH_TEST = "./data/discrate/"

def loadDisCrateSet(path):
    images = []
    labels = []
    for i,img in  enumerate(os.listdir(PATH_TEST)):
        if img[-3:] in ['jpg']:
            print(">>>reading image",img)
            npimg = plt.imread(PATH_TEST+img).reshape(784,4)
            npimg = np.asarray([  np.average(np.sum(x[:3])) for x in npimg ]).reshape(784)
            images.append(npimg)
            labels.append(int(img[:1]))
    return np.asarray(labels).reshape(len(labels),1),np.asarray(images)