import pandas as pd
from sklearn import svm
from sklearn.externals import joblib

PATH_TEST = "../data/test.csv"

# load test data and predict
print(">>>loading data...")
test_data=pd.read_csv(PATH_TEST)
test_data[test_data>0]=1

# load model
print(">>>loading model...")
clf = joblib.load('svm_svc.model') 


print(">>>predicting...")
results=clf.predict(test_data)

# output result
print(">>>saving results...")
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
print(">>>done.")
