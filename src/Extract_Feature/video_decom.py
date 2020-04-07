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
		FLANN_INDEX_LSH = 6
		index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 12, key_size = 20, multi_probe_level = 1)
		search_params = dict(checks=50)
		#self.flann = cv.FlannBasedMatcher(index_params, search_params)
		self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
		signal.signal(signal.SIGINT, self.exit_program)
		self.prev_des = None

	def exit_program(self, *args): # put in slam
		log.info("Exiting the program!")
		#self.boostfps.stopStream()
		os.system('pkill -9 python')	

	def detectCornerCombo(self, img):
		# extract
		kp = self.fast.detect(img, mask=None)
		kp = [cv.KeyPoint(x=kps.pt[0], y=kps.pt[1], _size=20) for kps in kp] # adjust _size
		# describe
		kp, des = self.orb.compute(img, kp)
		# match
		if self.prev_des is not None:
			#matches = self.flann.knnMatch(self.prev_des, orb_des, k=2) # zabije!
			matches = self.bf.match(des, self.prev_des) # zabije!
		self.prev_des = des			
		img = cv.drawKeypoints(img, keypoints=kp, outImage=None, color=(255,0,0))
		kp = np.array([(kps.pt[0], kps.pt[1]) for kps in kp])
		# create a dicitonary that this function returns - keypoints, description, matches

		return img



	def process_frame(self, img):
		processed_img = self.detectCornerCombo(img)
		cv.imshow('image', img) # put in fps booster?
		cv.imshow('filtered', processed_img)
		cv.waitKey(1)


