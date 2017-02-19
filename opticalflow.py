import sys
sys.path.append('/usr/local/Cellar/opencv3/3.2.0/lib/python2.7/site-packages')
import cv2
import numpy as np
cap = cv2.VideoCapture("/Users/lucastecot/Development/opticalflow/nits.mp4")
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    grey = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(next,100,200)
    cleanImg = cv2.multiply(grey, edges)
    cv2.imshow('frame2',grey)
    cv2.waitKey(0)
    cv2.imshow('frame2',edges)
    cv2.waitKey(0)
    cv2.imshow('frame2',cleanImg)
    cv2.waitKey(0)
    prvs = next
cap.release()
cv2.destroyAllWindows()