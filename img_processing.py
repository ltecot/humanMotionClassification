import sys
#sys.path.append('/usr/local/Cellar/opencv3/3.2.0/lib/python2.7/site-packages')
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages")
import cv2
import numpy as np
import os
import random

def show_image(im):
    height, width = im.shape[:2]
    res = cv2.resize(im,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
    cv2.imshow("Image", res)

def show_imageOrig(im):
    height, width = im.shape[:2]
    res = cv2.resize(im,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
    cv2.imshow("ImageOrig", res)

#kernel = np.ones((3,3),np.uint8)

#cap = cv2.VideoCapture("dogs.mp4")
#ret, frame1 = cap.read()
#prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#hsv = np.zeros_like(frame1)
#hsv[...,1] = 255

#fgbg = cv2.createBackgroundSubtractorMOG2(50, 16, False)  

#fgbg.setBackgroundRatio(0.8) # frames before object becomes foreground
#fgbg.setVarInit(500) # speed of adaption of new components

#i = 0
#while(1):
#    ret, frame2 = cap.read()
#    ret, frame2 = cap.read()    
#    frame2 = cv2.GaussianBlur(frame2,(9,9),0)
    
#    fgmask = fgbg.apply(frame2)
#    fgmask = fgbg.apply(frame2,fgmask, 0)
    #fgmask = cv2.dilate(fgmask,kernel,iterations = 5)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    '''
    ^^ Line above may or may not be good
    '''
    #if (i > 10 and i % 2 == 0):
        #cv2.imwrite(str(i) + ".png",fgmask)
#    show_image(fgmask)
    
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break

    #i += 1
#cap.release()
#cv2.destroyAllWindows()

#errorCount = 0
np.random.seed(133)
numLabels = 101
image_size_x = 240
image_size_y = 320
dataRoot = "./UCF-101/"

def processFolder(folder):
  #tick = 0
  #global errorCount
  print(dataRoot + folder)
  try:
    videoFileNames = os.listdir(dataRoot + folder)
  except:
    print("Not a directory, moving along.")
    return None, None
  #i = 0
  #data = np.zeros(shape=(len(videoFileNames)*1, image_size_x, image_size_y), dtype=np.float32)
  #labels = np.zeros(shape=(len(videoFileNames)*1, 101), dtype=np.float32)
  for videoName in videoFileNames:
    #if tick < 2:
    #  tick = tick + 1
    #  continue
    #tick = 0

    if random.random() < 0.98:
      continue

    try:
      print(videoName)
      cap = cv2.VideoCapture(dataRoot + folder + "/" + videoName)
#ret, frame1 = cap.read()
#prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#hsv = np.zeros_like(frame1)
#hsv[...,1] = 255

      fgbg = cv2.createBackgroundSubtractorMOG2(50, 16, False)  

      fgbg.setBackgroundRatio(0.8) # frames before object becomes foreground
      fgbg.setVarInit(500) # speed of adaption of new components

      i = 0
      frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      while(cap.get(cv2.CAP_PROP_POS_FRAMES) < frames - 3):
        #ret, frame2 = cap.read()
        ret, frame2 = cap.read()
        if ret == False:
          continue
        show_imageOrig(frame2)
        frame2 = cv2.GaussianBlur(frame2,(9,9),0)
    
        fgmask = fgbg.apply(frame2)
        fgmask = fgbg.apply(frame2,fgmask, 0)
        show_image(fgmask)
    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
          break
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
  #return data, labels

def iterData(folder):
  labelNames = os.listdir(folder)
  for i in range(len(labelNames)):#len(labelNames)
    processFolder(labelNames[i])

iterData(dataRoot)