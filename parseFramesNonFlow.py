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
import time

errorCount = 0
np.random.seed(int(time.time()))
numLabels = 101
image_size_x = 240
image_size_y = 320
dataRoot = "./UCF-101/"

def extractData(folder):
  #tick = 0
  global errorCount
  try:
    videoFileNames = os.listdir(dataRoot + folder)
  except:
  	print("Not a directory, moving along.")
  	return None, None
  i = 0
  for videoName in videoFileNames:
    z = 10
    #try:
    if videoName.endswith(".avi"):
      #i = 0
      cap = cv2.VideoCapture(dataRoot + folder + "/" + videoName)
      frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      #i = 0
      #print(frames)
      if frames == 0:
        errorCount += 1
        continue
      print(videoName)
      print(frames)
      for _ in range(10):
        #print("went wrong here")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames * random.random()) % frames)
        ret, frame = cap.read()
        #print("went wrong here")
        try:
          frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        except:
          errorCount += 1
          continue
        if ret == False:
          errorCount += 1
          raise Exception("Couldn't read image")
        if frame.shape != (image_size_x, image_size_y):
          #raise Exception('Unexpected image shape: %s' % str(frame.shape))
          print('Unexpected image shape: %s' % str(frame.shape))
          errorCount = errorCount + 1
          continue
        #print("went wrong here")
        im = np.ndarray(shape=(image_size_x, image_size_y), dtype=np.int)
        for x in range(image_size_x):
          for y in range(image_size_y):
            im[x][y] = (frame[x][y] - 255/2)
        #print("went wrong here")
        cv2.imwrite(dataRoot + folder + "/" + "frame%d_%d.png" % (i,z) , im) 
        #data[i] = im
        #cv2.imshow('frame', im)
        #cv2.waitKey(0)
        #labels[i][index] = 1
        print("frame_", i, "_", z)
        z = z + 1
      i = i + 1
    #except:
    #  print('Something went wrong bro, skippin dat sheit')
  #return data, labels

def compileData(folder):
  labelNames = os.listdir(folder)
  dataArray = []
  labelsArray = []
  #print(len(labelNames))
  print(labelNames)
  ind = 0
  for i in range(len(labelNames)):#len(labelNames)
    #if i < 6:
    #  continue
    extractData(labelNames[i])

compileData(dataRoot)
