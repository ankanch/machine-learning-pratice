# this script will create more data based on given data 
# via following transformations: rotate, shift position

from scipy.ndimage import interpolation as itp
from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd

PATH_TRAIN = "../data/train.csv"

# loading training data
print(">>>loading data...")
labeled_images = pd.read_csv(PATH_TRAIN)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
images = np.asarray([ x.reshape(28,28) for x in images.as_matrix() ])
labels = labels.as_matrix()
lenx = labels.size
i = 0

# start transformation
# rotate from 45, to 180, with 45 degree per step
rotate_list = []
if True:
    rl = [ int((x+1)*15) for x in range(6)]
    rrl  = [ -1*x for x in rl]
    rl.extend(rrl)
    print(">>>rotating degrees:",rl,"\n>>>start rotating")
    for image,label in zip(images,labels):
        rotate_list.append([ label, image ])
        for r  in rl:
            print(">>>processing rotation",i,"/",lenx,",rotating",r," degree",end='\r')
            img = itp.rotate(image,r,reshape=False)
            rotate_list.append([ label, img ])
        i+=1
    print(">>>processing rotation",i,"/",lenx)

# random noisy
random_noisy_list = []
if True:
    i = 0
    for image,label in zip(images,labels):
        print(">>>processing random noisy",i,"/",lenx,end='\r')
        pick = random.randrange(75)
        #plt.subplot(211)
        #plt.title(label)
        #plt.imshow(image)
        for i in range(pick):
            lx = random.randrange(28)
            ly = random.randrange(28)
            image[lx][ly] = abs(image[lx][ly] - 255)
        #plt.subplot(212)
        #plt.imshow(image)
        #plt.show()
        i+=1
    print(">>>processing random noisy",i,"/",lenx)


del images
del labels
# combine these three transformation dataset with original dataset
print(">>>Conbining dataset...")
new_list = []
new_list.extend(rotate_list)
new_list.extend(random_noisy_list)
new_list = [ [ x[0],x[1].reshape(1,784) ] for x in new_list ]

print(">>>preprocessing...")
final_str = []
label_str = ",".join(list(labeled_images))
del labeled_images
i = 0
lenx = len(new_list)
for label,image in new_list:
    print(">>>converting to str",i,"/",lenx,end='\r')
    istr = ",".join( [ str(x) for x in image[0] ] )
    istr = str(label[0]) + "," + istr
    final_str.append(istr)
    i+=1
print(">>>converting to str",i,"/",lenx)
print(">>>saving...")
res = "\n".join(final_str)
res = label_str + "\n" + res
with open("new_dataset.csv","w") as ff:
    ff.write(res)
print(">>>all done.")