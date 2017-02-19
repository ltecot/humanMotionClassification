from __future__ import print_function
import sys
#sys.path.append('/usr/local/Cellar/opencv3/3.2.0/lib/python2.7/site-packages')
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages")
import cv2
import random
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

image_size_x = 240
image_size_y = 320
num_labels = 101
num_channels = 1 # grayscale

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_samples = 3

graph = tf.Graph()

with graph.as_default():

  # Input data.
  cap = cv2.VideoCapture(sys.argv[0])
  data = np.zeros(shape=(num_samples, image_size_x, image_size_y), dtype=np.float32)
  frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(num_samples):
      cap.set(cv2.CAP_PROP_POS_FRAMES, frames * random.random())
      ret, frame = cap.read()
      frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      if ret == False:
        raise Exception("Couldn't read image")
      if frame.shape != (image_size_x, image_size_y):
        raise Exception('Unexpected image shape: %s' % str(frame.shape))
      im = np.ndarray(shape=(image_size_x, image_size_y), dtype=np.float32)
      for x in range(image_size_x):
        for y in range(image_size_y):
          im[x][y] = (frame[x][y].astype(float) - 255.0/2) / 255.0
      data[i] = im

  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1), name = "l1w")
  layer1_biases = tf.Variable(tf.zeros([depth]), name = "l1b")
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1), name = "l2w")
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]), name = "l2b")
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size_x // 4 * image_size_y // 4 * depth, num_hidden], stddev=0.1), name = "l3w")
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name = "l3b")
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1), name = "l4w")
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name = "l4b")
  
  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
  saver.restore(sess, "model.ckpt")
  print("Model restored.")
  prediction = tf.nn.softmax(model(data))
  for pred in prediction:
  	print(np.argmax(pred))
