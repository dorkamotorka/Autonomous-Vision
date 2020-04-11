import os
import sys
sys.path.append('..')
from logger import Logger
import numpy as np
import signal
import cv2 as cv
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.namedWindow('filtered', cv.WINDOW_NORMAL)

log = Logger('Feats_Extract')

class FeatureExtract(object):
	def __init__(self):
		self.fast = cv.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16) 	
		self.orb = cv.ORB_create(nfeatures=700, scaleFactor=1.5, nlevels=3, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
		self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
		signal.signal(signal.SIGINT, self.exit_program)
		self.last = None
		self.filter_img = None

	def exit_program(self, *args): # put in slam
		log.info("Exiting the program!")
		os.system('pkill -9 python')	

	def detectCombo(self, img):
		# extract
		kp = self.fast.detect(img, mask=None)
		kp = [cv.KeyPoint(x=kps.pt[0], y=kps.pt[1], _size=20) for kps in kp] # adjust _size
		# describe
		kp, des = self.orb.compute(img, kp)
		# match
		good, matches = [], []
		if self.last is not None:
			matches = self.bf.knnMatch(des, self.last['des'], k=2)
			# Lowe's ratio
			for m,n in matches:
				if m.distance < 0.75*n.distance:
					kp1 = kp[m.queryIdx].pt
					kp2 = self.last['kps'][m.trainIdx].pt
					good.append((kp1, kp2))

		if len(good) > 0:
			good = np.array(good)	
			# RANSAC	
			model, inliers = ransac((good[:, 0], good[:, 1]),
						FundamentalMatrixTransform,
						min_samples=8,
						residual_threshold=1,
						max_trials=100)
			print(inliers)
			good = good[inliers]
			
		self.last = {'kps': kp, 'des': des}			
		self.filter_img = cv.drawKeypoints(img, keypoints=kp, outImage=None, color=(255,0,0))
		kp = np.array([(kps.pt[0], kps.pt[1]) for kps in kp])

		return kp, des, matches


	def process_frame(self, img):
		cv.imshow('image', img)
		if self.filter_img is not None:
			cv.imshow('filtered', self.filter_img)
		cv.waitKey(1)
	
