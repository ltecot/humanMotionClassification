from __future__ import print_function
import sys
sys.path.append('/usr/local/Cellar/opencv3/3.2.0/lib/python2.7/site-packages')
import cv2
import random
import numpy as np
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

#help('cv2')
cap = cv2.VideoCapture("UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi", 0)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT )
cap.set(cv2.CAP_PROP_POS_FRAMES, frames * random.random())
ret, frame = cap.read()
#print(ret)
frameGscl = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
cv2.imshow('frame', frameGscl)
cv2.waitKey(0)