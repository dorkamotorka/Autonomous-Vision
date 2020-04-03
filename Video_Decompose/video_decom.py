import numpy as np
import cv2 as cv
import imutils


cv.namedWindow('image', cv.WINDOW_NORMAL)

def process_frame(img):
	cv.imshow('image', img)
	cv.waitKey(10)

if __name__ == '__main__':
	capt = cv.VideoCapture('test_countryroad.mp4')

	while capt.isOpened():
		retval, frame = capt.read()
		if retval == True:
			process_frame(frame)
		else:
			break
	
	capt.release()
	cv.destroyAllWindows()
