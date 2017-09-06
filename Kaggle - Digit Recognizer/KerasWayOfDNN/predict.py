from keras.models import load_model
import pandas as pd

PATH_TEST = "../data/test.csv"

# load test data
print('>>>loading test data...')
test_data=pd.read_csv(PATH_TEST)
test_data[test_data>0]=1
test_data = test_data.as_matrix().reshape(test_data.shape[0], 28, 28,1)

# load model
print('>>>loading model...')
model = load_model('digital_recog_w_sequentialDenseNN.h5fmodel')

# eval test data
print('evaluating test data...')
results = model.predict(test_data,batch_size=128)
results = [ x.tolist().index(max(x)) for x in results]

# output result
print(">>>saving results...")
df = pd.DataFrame({'Label':results})
df.index += 1
df.index.name='ImageId'
df.to_csv('results.csv')
print(">>>done.")