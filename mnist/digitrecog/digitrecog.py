import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

from PIL import Image, ImageOps
import numpy as np

def load_image(filename):
    img = Image.open(filename)
    img.load()
    img = ImageOps.invert(img)
    img = img.convert("L")
    img = img.resize((28, 28), Image.ANTIALIAS)
    data = np.asarray( img.getdata() )
    data = data.reshape( -1, 28, 28, 1)[0]
    return data


X, Y, test_x, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

convnet = input_data(shape=[None, 28, 28, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

model = tflearn.DNN(convnet)

'''
model.fit(X, Y, n_epoch=10, validation_set=(test_x, test_y), snapshot_step=500, show_metric=True, run_id='mnist')
model.save('models/tflearncnn.model')
'''

model.load('models/tflearncnn.model');

print(model.predict([test_x[1]]))
print(model.predict([load_image('sampleimages/three_1.jpg')]))
print(model.predict([load_image('sampleimages/seven_1.jpg')]))
print(model.predict([load_image('sampleimages/seven_2.jpg')]))
print(model.predict([load_image('sampleimages/two_1.jpg')]))
print(model.predict([load_image('sampleimages/two_2.jpg')]))
print("expected 5 --", model.predict([load_image('sampleimages/five_1.jpg')]))