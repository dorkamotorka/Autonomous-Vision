import numpy as np
import cv2 as cv
import imutils

cap = cv.VideoCapture('test_drone.mp4')
if not cap.isOpened():
    	print("Cannot open file video")
    	exit()

while cap.isOpened():
	ret, frame = cap.read()
	# if frame is read correctly ret is True
	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break

	frame = imutils.resize(frame, width=920)
	#gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	#cv.imshow('frame', gray)
	cv.imshow('frame', frame)
	if cv.waitKey(1) == ord('q'):
		break

cap.release()
cv.destroyAllWindows()
