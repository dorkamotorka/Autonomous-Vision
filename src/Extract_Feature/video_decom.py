import sys
sys.path.append('..')
from logger import Logger
import os
import numpy as np
import signal
import cv2 as cv
import imutils
from fps_thread import BoostedFPS

# FEATURE EXTRACTOR THREAD!

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.namedWindow('filtered', cv.WINDOW_NORMAL)

log = Logger('Video_decom')

class VideoCV(object):
	def __init__(self):
		self.frame = None
		self.boostfps = BoostedFPS('test_countryroad.mp4')
		self.bstream = self.boostfps.startStream()
		# FAST Algorithm
		self.fast = cv.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16) 
		self.fast_kp = None
		# ORB Algorithm		
		self.orb = cv.ORB_create(nfeatures=700, scaleFactor=1.5, nlevels=3, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
		self.orb_kp = None
		self.orb_despr = None
		# COMBO Algorithm(FAST extractor + ORB descriptor)
		self.combo_kp = None
		self.combo_descr = None
		# keypoints and description
		self.features = None

	def exit_program(self, *args):
		log.info("Exiting the program!")
		self.bstream.stopStream()
		os.system('pkill -9 python')	

	def detectCornerCombo(self, img):
		# get keypoints
		self.combo_kp = self.fast.detect(img, mask=None)
		# describe keypoints
		self.combo_kp, self.orb_descr = self.orb.compute(img, self.combo_kp)

	def detectCornerFAST(self, img):
		self.fast_kp = self.fast.detect(img, mask=None)
		img = cv.drawKeypoints(img, keypoints=self.fast_kp, outImage=None, color=(255,0,0))		

		return img

	def detectCornerORB(self, img):
		self.orb_kp = self.orb.detect(img,mask=None)
		self.orb_kp, self.orb_descr = self.orb.compute(img, self.orb_kp)
		img = cv.drawKeypoints(img, keypoints=self.orb_kp, outImage=None, color=(255,0,0), flags=0)
		return img

	def process_frame(self, img):
		processed_img = self.detectCornerFAST(img)
		cv.imshow('image', img)
		cv.imshow('filtered', processed_img)
		cv.waitKey(1)


if __name__ == '__main__':
	videocv = VideoCV()
	signal.signal(signal.SIGINT, videocv.exit_program)
	start = cv.getTickCount()	

	running = True
	while running:
		videocv.frame = videocv.bstream.getFrame()
		videocv.process_frame(videocv.frame)
		running = videocv.bstream.checkBuffer()

	end = cv.getTickCount()		
	exec_time = (end - start)/ cv.getTickFrequency()
	log.info(f"Execution time: {exec_time} seconds")
	cv.destroyAllWindows()


#colorspace_flags = [i for i in dir(cv) if i.startswith('COLOR_')]
#print(flags)

#check HSV values for colors
#green = np.uint8([[[0,255,0 ]]])
#hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
#print( hsv_green )

#white
#lower_thresh = np.array([0,0,200])
#upper_thresh = np.array([180,50,255])
