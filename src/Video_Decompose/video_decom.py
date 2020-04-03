import numpy as np
import signal
import cv2 as cv
import imutils
from fps_thread import BoostedFPS
import os


cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.namedWindow('filtered', cv.WINDOW_NORMAL)

class VideoCV(object):
	def __init__(self):
		self.frame = None
		self.boostfps = BoostedFPS('test_countryroad.mp4')
				
		self.init_video()

	def init_video(self):
		self.bstream = self.boostfps.startStream()

	def exit_program(self, *args):
		print("Exiting the program!")
		self.bstream.stopStream()
		os.system('pkill -9 python')	

def extractCorner(img):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	extracted = cv.cornerHarris(gray, 2, 11, 0.04)

	return extracted

def threshHSV(img):
	img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	lower_thresh = np.array([0,50,50])
	upper_thresh = np.array([50,255,255])
	threshed = cv.inRange(img_hsv, lower_thresh, upper_thresh)
	konjugated = cv.bitwise_and(img, img, mask=threshed)
	return konjugated
	
def process_frame(img):
	try:
		processed_img = extractCorner(img)
		cv.imshow('image', img)
		cv.imshow('filtered', processed_img)
		cv.waitKey(1)
	except:
		pass

if __name__ == '__main__':
	videocv = VideoCV()
	signal.signal(signal.SIGINT, videocv.exit_program)
	
	running = True
	while running:
		videocv.frame = videocv.bstream.getFrame()
		process_frame(videocv.frame)
		running = videocv.bstream.checkBuffer()		

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
