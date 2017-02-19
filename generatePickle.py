from __future__ import print_function
import sys
sys.path.append('/usr/local/Cellar/opencv3/3.2.0/lib/python2.7/site-packages')
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages")
import cv2
import random
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os

errorCount = 0
np.random.seed(133)
numLabels = 101
image_size_x = 240
image_size_y = 320
dataRoot = "./UCF-101/"

def extractData(folder, index):
  #tick = 0
  global errorCount
  try:
    videoFileNames = os.listdir(dataRoot + folder)
  except:
  	print("Not a directory, moving along.")
  	return None, None
  i = 0
  data = np.zeros(shape=(len(videoFileNames)*1/30, image_size_x, image_size_y), dtype=np.float32)
  labels = np.zeros(shape=(len(videoFileNames)*1/30, 101), dtype=np.float32)
  for videoName in videoFileNames:
    #if tick < 30:
    #  tick = tick + 1
    #  continue
    #tick = 0
    try:
      cap = cv2.VideoCapture(dataRoot + folder + "/" + videoName)
      frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      #i = 0
      print(frames)
      for _ in range(1):
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

def pickleData(folder):
  dataset, labels = compileData(folder)

  dataset, labels = randomize(dataset, labels)

  #print(dataset.shape)
  #print(len(dataset))


  validStart = int(len(dataset) * 0.9)
  #print(validStart)
  testStart = int(len(dataset) * 0.95)
  train_data = dataset[:validStart]
  train_labels = labels[:validStart]
  valid_data = dataset[validStart:testStart]
  valid_labels = labels[validStart:testStart]
  test_data = dataset[testStart:]
  test_labels = labels[testStart:]

  print(dataset.shape)
  print(labels.shape)
  print(train_data.shape)
  print(train_labels.shape)
  print(valid_data.shape)
  print(valid_labels.shape)
  print(test_data.shape)
  print(test_labels.shape)
  print(errorCount)

  train_data, train_labels = randomize(train_data, train_labels)
  valid_data, valid_labels = randomize(valid_data, valid_labels)
  test_data, test_labels = randomize(test_data, test_labels)

  #testing
  #for i in range(train_data.shape[0]):
  #  print(train_data[i])
  #  cv2.imshow('frame', train_data[i])
  #  print(np.argmax(train_labels[i]))
  #  cv2.waitKey(0)

  #print("VALID")

  #for i in range(valid_data.shape[0]):
  #  print(valid_data[i])
  #  cv2.imshow('frame', valid_data[i])
  #  print(np.argmax(valid_labels[i]))
  #  cv2.waitKey(0)

  #print("TEST")

  #for i in range(test_data.shape[0]):
  #  print(test_data[i])
  #  cv2.imshow('frame', test_data[i])
  #  print(np.argmax(test_labels[i]))
  #  cv2.waitKey(0)

  pickle_file = os.path.join(folder, 'action.pickle')
  try:
    f = open(pickle_file, 'wb')
    save = {
      'train_dataset': train_data,
      'train_labels': train_labels,
      'valid_dataset': valid_data,
      'valid_labels': valid_labels,
      'test_dataset': test_data,
      'test_labels': test_labels,
      }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

pickleData(dataRoot)

#cap = cv2.VideoCapture("UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi", 0)
#frames = cap.get(cv2.CAP_PROP_FRAME_COUNT )
#cap.set(cv2.CAP_PROP_POS_FRAMES, frames * random.random())
#ret, frame = cap.read()
#frameGscl = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#cv2.imshow('frame', frameGscl)
#cv2.waitKey(0)