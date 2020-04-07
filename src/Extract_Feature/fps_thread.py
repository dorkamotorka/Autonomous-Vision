import sys
sys.path.append('..')
from logger import Logger
from threading import Thread
from queue import Queue
import cv2 as cv

log = Logger('Fps_thread')

class BoostedFPS:
	def __init__(self, src=0, queueSize=1000):
		'''Video stream thread setup'''
		self.src = src		
		self.Q = Queue(maxsize=queueSize) # frames queue
		self.stopped = False
		Thread(target=self.VideoStream, args=(self.stopped,), daemon=True).start()

	def VideoStream(self, stop):
		stream =  cv.VideoCapture(self.src)
		while not stop:
			_, frame = stream.read()
			self.Q.put(frame)		
	
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
