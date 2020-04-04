import sys
sys.path.append('..')
from logger import Logger
from threading import Thread
from queue import Queue
import cv2 as cv

log = Logger('Fps_thread')

class BoostedFPS:
	def __init__(self, src=0, queueSize=10000):
		'''Video stream thread setup'''		
		self.stream =  cv.VideoCapture(src)
		self.frame = None
		self.stopped = False
		self.Q = Queue(maxsize=queueSize) # frames queue

	def startStream(self):
		Thread(target=self.VideoStream, daemon=True).start()
		return self

	def VideoStream(self):
		while not self.stopped:
			_, self.frame = self.stream.read()
			self.Q.put(self.frame)
				
	def getFrame(self):
		return self.Q.get();

	def checkBuffer(self):
		if self.Q.full():
			log.error("Frame Queue full!")
			return False
		else:
			return self.Q.qsize() > 0

	def stopStream(self):
		self.stopped = True
