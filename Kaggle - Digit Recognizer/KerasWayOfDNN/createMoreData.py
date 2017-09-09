# this script will create more data based on given data 
# via following transformations: rotate, shift position

from scipy.ndimage import interpolation as itp
from matplotlib import pyplot as plt
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
if False:
    for image,label in zip(images,labels):
        print(">>>processing rotation",i,"/",lenx,end='\r')
        img45 = itp.rotate(image,45,reshape=False)
        img90 = itp.rotate(image,90,reshape=False)
        img135 = itp.rotate(image,135,reshape=False)
        img180 = itp.rotate(image,180,reshape=False)
        rotate_list.append([ label, image ])
        rotate_list.append([ label, img45 ])
        rotate_list.append([ label, img90 ])
        rotate_list.append([ label, img135 ])
        rotate_list.append([ label, img180 ])
        i+=1
    print(">>>processing rotation",i,"/",lenx)

# shift
shift_list = []
i = 0
for image,label in zip(images,labels):
    print(">>>processing shifting",i,"/",lenx,end='\r')
    shift_rb5 = itp.shift(image,[3,3])
    shift_lt5 = itp.shift(image,[-3,-3])
    shift_rt5 = itp.shift(image,[3,-3])
    shift_lb5 = itp.shift(image,[-3,3])
    shift_list.append( [ label, shift_rb5 ] )
    shift_list.append( [ label, shift_lt5 ] )
    shift_list.append( [ label, shift_rt5 ] )
    shift_list.append( [ label, shift_lb5 ] )
    i+=1
print(">>>processing shifting",i,"/",lenx)


# combine these three transformation dataset with original dataset
print(">>>Conbining dataset...")
new_list = []
new_list.extend(rotate_list)
new_list.extend(shift_list)
new_list = [ [ x[0],x[1].reshape(1,784) ] for x in new_list ]

print(">>>preprocessing...")
final_str = []
label_str = ",".join(list(labeled_images))
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
with open("new_dataset_shift_only.csv","w") as ff:
    ff.write(res)
print(">>>all done.")