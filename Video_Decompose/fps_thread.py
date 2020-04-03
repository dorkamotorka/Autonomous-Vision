from threading import Thread
import time
import cv2 as cv
import os

class BoostedFPS:
	def __init__(self, src=0):
		'''Video stream thread setup'''
		self.stream =  cv.VideoCapture(src)
		self.retval = None
		self.frame = None
		self.stopped = False

	def startStream(self):
		Thread(target=self.VideoStream).start()
		return self

	def VideoStream(self):
		while not self.stopped:
			_, self.frame = self.stream.read()
		
	def getFrame(self):
		return self.frame;

	def stopStream(self):
		self.stopped = True
