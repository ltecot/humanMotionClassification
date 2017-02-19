import sys
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages")
import cv2
import numpy as np

def show_image(im):
    cv2.imshow("Image", im)
    
kernel = np.ones((53,53),np.uint8)

cap = cv2.VideoCapture("UCF-101/RopeClimbing/v_RopeClimbing_g01_c01.avi")
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

fgbg = cv2.createBackgroundSubtractorMOG2(50, 16, False) #history, 
fgbg.setBackgroundRatio(0.8) # time before it object becomes foreground
#fgbg.setVarThresholdGen(16) # higher value = less components
fgbg.setVarInit(500) # speed of adaption of new components
fgbg.setDetectShadows(False)

#try to use outline optical flow thing + this

#i = 0
while(1):
    ret, frame2 = cap.read()
    frame2 = cv2.GaussianBlur(frame2,(9,9),0)
    
    # 1 Threshold I tried
    #_, threshold_img = cv2.threshold(frame2, 150, 180 , cv2.THRESH_BINARY)
    #show_image(threshold_img)



    fgmask = fgbg.apply(frame2)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #show_image(frame2)
    show_image(fgmask)

    '''
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    

    # remove noise
    img = cv2.GaussianBlur(next,(3,3),0)
    

    # convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    #sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    #sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
    '''

    '''
    flow = cv2.calcOpticalFlowFarneback(
        prvs,next, None, 0.5, 3, 10, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',bgr)
    '''

    

    '''
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
    '''
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    '''
    prvs = next
    '''
    #i += 1
cap.release()
cv2.destroyAllWindows()
