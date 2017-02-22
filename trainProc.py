from __future__ import print_function
import sys
#sys.path.append('/usr/local/Cellar/opencv3/3.2.0/lib/python2.7/site-packages')
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages")
import cv2
import random
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
import tensorflow as tf

errorCount = 0
np.random.seed(133)
numLabels = 101
image_size_x = 240
image_size_y = 320
dataRoot = "./UCF-101/"

def procImage(frame):
  #frame1 = cap.read()
  #prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
  #hsv = np.zeros_like(frame1)
  #hsv[...,1] = 255
  fgbg = cv2.createBackgroundSubtractorMOG2(50, 16, False) #history, 
  #Threshold on the squared Mahalanobis distance, and shadow detection bool 
  fgbg.setBackgroundRatio(0.8) # frames before object becomes foreground
  fgbg.setVarInit(500) # speed of adaption of new components
  #ret, frame2 = cap.read()
  #ret, frame2 = cap.read()    
  frame2 = cv2.GaussianBlur(frame,(9,9),0)
  fgmask = fgbg.apply(frame)
  fgmask = fgbg.apply(frame,fgmask, 0)
  return frame

def extractData(folder, index):
  #tick = 0
  global errorCount
  try:
    videoFileNames = os.listdir(dataRoot + folder)
  except:
  	print("Not a directory, moving along.")
  	return None, None
  i = 0
  data = np.zeros(shape=(len(videoFileNames)*1, image_size_x, image_size_y), dtype=np.float32)
  labels = np.zeros(shape=(len(videoFileNames)*1, 101), dtype=np.float32)
  for videoName in videoFileNames:
    #if tick < 2:
    #  tick = tick + 1
    #  continue
    #tick = 0
    try:
      cap = cv2.VideoCapture(dataRoot + folder + "/" + videoName)
      frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      #i = 0
      print(frames)
      for _ in range(1):
        if frames == 0:
          continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames * random.random()) % frames)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret == False:
          raise Exception("Couldn't read image")
        print(videoName)
        if frame.shape != (image_size_x, image_size_y):
          #raise Exception('Unexpected image shape: %s' % str(frame.shape))
          print('Unexpected image shape: %s' % str(frame.shape))
          errorCount = errorCount + 1
          continue
        frame = procImage(frame)
        im = np.ndarray(shape=(image_size_x, image_size_y), dtype=np.float32)
        for x in range(image_size_x):
          for y in range(image_size_y):
            im[x][y] = (frame[x][y].astype(float) - 255.0/2) / 255.0
        data[i] = im
        #cv2.imshow('frame', data[i])
        #cv2.waitKey(0)
        labels[i][index] = 1
        i = i + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
  return data, labels

def compileData(folder):
  labelNames = os.listdir(folder)
  dataArray = []
  labelsArray = []
  #print(len(labelNames))
  print(labelNames)
  ind = 0
  for i in range(len(labelNames)):#len(labelNames)
    data, labels = extractData(labelNames[i], ind)
    #print(data)
    #print(labels)
    if data != None and labels != None:
      ind = ind + 1
      for z in range(len(data)):
      	dataArray.append(data[z])
      	labelsArray.append(labels[z])

  #testing
  #print(ind)
  #for i in range(len(dataArray)):
    #print(dataset[i])
    #cv2.imshow('frame', dataArray[i])
    #print(np.argmax(labelsArray[i]))
    #cv2.waitKey(1)

  return np.array(dataArray), np.array(labelsArray)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

dataset, labels = compileData(dataRoot)

dataset, labels = randomize(dataset, labels)

  #print(dataset.shape)
  #print(len(dataset))


validStart = int(len(dataset) * 0.9)
#print(validStart)
testStart = int(len(dataset) * 0.95)
train_dataset = dataset[:validStart]
train_labels = labels[:validStart]
valid_dataset = dataset[validStart:testStart]
valid_labels = labels[validStart:testStart]
test_dataset = dataset[testStart:]
test_labels = labels[testStart:]

train_dataset, train_labels = randomize(train_dataset, train_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

image_size_x = 240
image_size_y = 320
num_labels = 101
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size_x, image_size_y, num_channels)).astype(np.float32)
  #labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size_x, image_size_y, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
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
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))

  num_steps = 1001
  saver = tf.train.Saver()
  #init_op = tf.global_variables_initializer()

with tf.Session(graph=graph) as session:
  #tf.global_variables_initializer().run()
  #tf.initialize_all_variables()
  init = tf.global_variables_initializer()
  session.run(init)
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  save_path = saver.save(session, "model.ckpt")
  print("Model saved in file: %s" % save_path)