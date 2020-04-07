# main program for all classes
from logger import Logger
import sys
sys.path.append('Extract_Feature')
from qbooster import BoostedFPS
from feats_extra import FeatureExtract
import cv2 as cv

log = Logger('slam')

if __name__ == '__main__':
	feats = FeatureExtract()
	booster = BoostedFPS('test_countryroad.mp4')
	start = cv.getTickCount()	

	running = True
	while running:
		feats.frame = booster.getFrame()
		feats.process_frame(feats.frame)
		_kp, _des, _matches = feats.detectCombo(feats.frame)
		print(_kp)
		running = booster.checkBuffer()

	end = cv.getTickCount()		
	exec_time = (end - start)/ cv.getTickFrequency()
	log.info(f"Execution time: {exec_time} seconds")
	booster.stopStream()
	cv.destroyAllWindows()
