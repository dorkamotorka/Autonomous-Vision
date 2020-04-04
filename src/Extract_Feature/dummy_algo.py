# for functions not used 
def detectCornerShiTomasi(self, img): # Shi-Tomasi Corner Detector
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	corners = cv.goodFeaturesToTrack(gray, maxCorners=3000, qualityLevel=0.01, minDistance=4)
	for i in corners:
		x,y = i.ravel()
		cv.circle(img, (x,y), 3, 255, -1)

	return img

def threshHSV(self, img):
	img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	lower_thresh = np.array([0,50,50])
	upper_thresh = np.array([50,255,255])
	threshed = cv.inRange(img_hsv, lowerb=lower_thresh, upperb=upper_thresh)
	konjugated = cv.bitwise_and(img, img, mask=threshed)
	
	return konjugated
