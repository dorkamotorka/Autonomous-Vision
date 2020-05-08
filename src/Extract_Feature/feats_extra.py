import numpy as np
import cv2 as cv
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
#import utils.helper


class FeatureExtract(object):
    def __init__(self):
        self.fast = cv.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16) 	
        self.orb = cv.ORB_create(nfeatures=700, scaleFactor=1.5, nlevels=3, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        self.last = None
    
    def ParamsCallback(self, x):
        #print(x)
        pass

    def detectCombo(self, img):
        blur = cv.GaussianBlur(img, (5,5), 0)
        # extract
        kp = self.fast.detect(blur, mask=None)
        #kp = [cv.KeyPoint(x=kps.pt[0], y=kps.pt[1], _size=20) for kps in kp] # adjust _size
        # describe
        kp, des = self.orb.compute(blur, kp)
        # match
        good, matches = [], []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            matches = np.array(matches)
            print(f"MATCHES: {matches.shape[0]}")
            # Lowe's ratio
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kp[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    good.append((kp1, kp2))

        if len(good) > 0:
            good = np.array(good)
            #util.normalize(good)
            print(f"GOOD: {good.shape[0]}")	
            # RANSAC	
            model, inliers = ransac((good[:, 0], good[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)
            good = good[inliers]
            #print(good)
            
        self.last = {'kps': kp, 'des': des}			
        #kp = np.array([(kps.pt[0], kps.pt[1]) for kps in kp])

        return kp, des, matches


