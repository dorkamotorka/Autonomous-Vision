import numpy as np
import cv2 as cv
from threading import Thread

def nothing():
    pass

class DynamicConfigure(object):
    def __init__(self, window):
        self.win = window
        cv.createTrackbar('NumKeyPoints', self.win, 100, 1000, nothing)
        #cv.createTrackbar('NumKeyPoints', self.win, 100, 1000, nothing)
        #cv.createTrackbar('NumKeyPoints', self.win, 100, 1000, nothing)
        switch = '0 : OFF \n1 : ON'
        cv.createTrackbar(switch, self.win, 0,1, nothing)
        Thread(target=self.track, args=(), daemon=True).start() 



    def track(self, img):
        num = cv.getTrackbarPos('NumKeyPoints', self.win)
        toggle = cv.getTrackbarPos(switch, self.win)

        if toggle == 0:
            img[:] = 0
        else:
            img[:] = [num]
        
        return img



