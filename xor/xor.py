import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
from random import randint
import tensorflow as tf
import numpy as np;


template_x = [ [0, 0], [0, 1], [1, 0], [1, 1] ]
template_y = [ [0], [1], [1], [0] ]
inputs = np.array(template_x)
targets = np.array(template_y)

input_layer = input_data(shape=[None, 2])
convnet = fully_connected(input_layer, 16, activation='tanh')
#convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 1, activation='tanh')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='mean_square')

model = tflearn.DNN(convnet)
model.fit(inputs, targets, n_epoch=1000, show_metric=True, batch_size=100, shuffle=True, run_id='xor')
#model.fit(template_x, template_y)
#model.save('models/tflearncnn.model')
#model.load('models/tflearncnn.model');

print(model.predict( [ [1, 1], [0, 0], [1, 0], [0, 1] ] ))