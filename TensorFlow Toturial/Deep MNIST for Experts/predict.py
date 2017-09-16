import tensorflow as tf
import pandas as pd
import numpy as np

PATH_TEST = "../data/test.csv"

# load test data
print('>>>loading test data...')
test_data=pd.read_csv(PATH_TEST)
test_data /= 255
mean = np.mean(test_data)
test_data -= mean
test_data = np.asarray([ x.reshape(28,28,1) for x in test_data.as_matrix() ])
print(len(test_data))

results = None
with tf.Session() as sess:
    s = tf.saved_model.loader.load(sess,"CNN4mnist", "./model")
    y_conv = s.get_tensor_by_name("y_conv:0")
    results =  sess.run("y_conv", feed_dict={"x": test_data})

print(results)
print(">>>saving results...")
df = pd.DataFrame({'Label':results})
df.index += 1
df.index.name='ImageId'
df.to_csv('results.csv')