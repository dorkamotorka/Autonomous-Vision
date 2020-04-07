import os
import sys
sys.path.append('..')
from logger import Logger
import numpy as np
import signal
import cv2 as cv

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.namedWindow('filtered', cv.WINDOW_NORMAL)

log = Logger('Video_decom')

class FeatureExtract(object):
	def __init__(self):
		# COMBO Algorithm(FAST extractor + ORB descriptor)
		self.fast = cv.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16) 	
		self.orb = cv.ORB_create(nfeatures=700, scaleFactor=1.5, nlevels=3, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
		self.combo_kp = None
		self.combo_descr = None
		signal.signal(signal.SIGINT, self.exit_program)

	def exit_program(self, *args): # put in slam
		log.info("Exiting the program!")
		#self.boostfps.stopStream()
		os.system('pkill -9 python')	

	def detectCornerCombo(self, img):
		# extract
		self.combo_kp = self.fast.detect(img, mask=None)
		self.combo_kp = [cv.KeyPoint(x=kp.pt[0], y=kp.pt[1], _size=20) for kp in self.combo_kp] # zabije - adust _size
		# describe
		self.combo_kp, self.orb_descr = self.orb.compute(img, self.combo_kp)
		img = cv.drawKeypoints(img, keypoints=self.combo_kp, outImage=None, color=(255,0,0))
		self.combo_kp = np.array([(kp.pt[0], kp.pt[1]) for kp in self.combo_kp])

		return img



	def process_frame(self, img):
		processed_img = self.detectCornerCombo(img)
		cv.imshow('image', img) # put in fps booster?
		cv.imshow('filtered', processed_img)
		cv.waitKey(1)


