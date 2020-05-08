# main program for all classes
from logger import Logger
import signal
import sys
import os
sys.path.append('Extract_Feature')
from qbooster import BoostedFPS
from feats_extra import FeatureExtract
sys.path.append('GUI')
import cv2 as cv
import numpy as np


running = True
#trackbar = 'Configuration'
cv.namedWindow('Original', cv.WINDOW_NORMAL)
cv.namedWindow('Processed', cv.WINDOW_NORMAL)
#cv.namedWindow(trackbar, flags=cv.WINDOW_AUTOSIZE)

log = Logger('slam')

def main():
    slam = Slam()
    signal.signal(signal.SIGINT, slam.exit_program)
    start = cv.getTickCount()	

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
        cv.imshow('image', img) 
        #num = cv.getTrackbarPos('value_name', trackbar)
        #self.feats.ParamsCallback(num)
        _kp, _des, _matches = self.feats.detectCombo(img)
        cv.drawKeypoints(img, keypoints=_kp, outImage=img, color=(255,0,0))
        cv.imshow('Processed', img)
        cv.waitKey(1)


    def exit_program(self, *args):
        global running
        print("Exiting the program!")
        os.system('pkill -9 python')
        self.booster.stopStream()
        running = False


#put in main
if __name__ == '__main__':
    main()

