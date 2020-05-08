
def HomogenousCoord(pts):
    # [x,y] -> [x,y,1]
    pts = np.append(pts, np.ones([pts.shape[0], 1]), axis=1)# commented numpy array in feats_extract
    return pts

def normalize(self, norm_arr): # put in helpers
    maxpix = np.amax(norm_arr)
    norm_arr = norm_arr/maxpix # FIX?
    print(norm_arr)

def denormalize(self, denom_arr):
    pass		

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

def detectCornerFAST(self, img):
	self.fast_kp = self.fast.detect(img, mask=None)	
	img = cv.drawKeypoints(img, keypoints=self.fast_kp, outImage=None, color=(255,0,0))

	return img

def detectCornerORB(self, img):
	self.orb_kp = self.orb.detect(img,mask=None)
	self.orb_kp, self.orb_descr = self.orb.compute(img, self.orb_kp)
	img = cv.drawKeypoints(img, keypoints=self.orb_kp, outImage=None, color=(255,0,0), flags=0)

	return img

#colorspace_flags = [i for i in dir(cv) if i.startswith('COLOR_')]
#print(flags)

#check HSV values for colors
#green = np.uint8([[[0,255,0 ]]])
#hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
#print( hsv_green )
