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
		self.frame = None
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
		# PANGO DISPLAY
		#self.pango3d = Pango3D()
		signal.signal(signal.SIGINT, self.exit_program)

	def exit_program(self, *args):
		log.info("Exiting the program!")
		#self.boostfps.stopStream()
		os.system('pkill -9 python')	

	def detectCornerCombo(self, img):
		# extract
		self.combo_kp = self.fast.detect(img, mask=None)
		self.combo_kp = [cv.KeyPoint(x=kp.pt[0], y=kp.pt[1], _size=20) for kp in self.combo_kp] # zabije
		# description
		self.combo_kp, self.orb_descr = self.orb.compute(img, self.combo_kp)
		img = cv.drawKeypoints(img, keypoints=self.combo_kp, outImage=None, color=(255,0,0))
		self.combo_kp = np.array([(kp.pt[0], kp.pt[1]) for kp in self.combo_kp] ) # zabije

		return img

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
		processed_img = self.detectCornerCombo(img)
		cv.imshow('image', img)
		cv.imshow('filtered', processed_img)
		cv.waitKey(1)


