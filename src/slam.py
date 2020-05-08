# main program for all classes
import os
import sys
import signal
import cv2 as cv
import numpy as np
from utils.logger import Logger
from Extract_Feature.qbooster import BoostedFPS
from Extract_Feature.feats_extra import FeatureExtract


#trackbar = 'Configuration'
cv.namedWindow('Original', cv.WINDOW_NORMAL)
cv.namedWindow('Processed', cv.WINDOW_NORMAL)
#cv.namedWindow(trackbar, flags=cv.WINDOW_AUTOSIZE)

log = Logger('slam')

def main():
    slam = Slam()
    signal.signal(signal.SIGINT, slam.exit_program)
    start = cv.getTickCount()	
    running = True

    while running:
        frame = slam.booster.getFrame() # move
        slam.process_frame(frame)
        running = slam.booster.checkBuffer() # move

    end = cv.getTickCount()		
    exec_time = (end - start)/ cv.getTickFrequency()
    log.info(f"Execution time: {exec_time} seconds")
    cv.destroyAllWindows()



class Slam(object):
    def __init__(self):
        self.feats = FeatureExtract()
        self.booster = BoostedFPS('test_countryroad.mp4')
        #cv.createTrackbar('NumKeyPoints', trackbar, 100, 1000, lambda nothing : nothing)

    def process_frame(self, img):
        cv.imshow('Original', img) 
        #num = cv.getTrackbarPos('value_name', trackbar)
        #self.feats.ParamsCallback(num)
        _kp, _des, _matches = self.feats.detectCombo(img)
        cv.drawKeypoints(img, keypoints=_kp, outImage=img, color=(255,0,0))
        cv.imshow('Processed', img)
        cv.waitKey(1)


    def exit_program(self, *args):
        print("Exiting the program!")
        os.system('pkill -9 python')
        self.booster.stopStream()


#put in main
if __name__ == '__main__':
    main()

