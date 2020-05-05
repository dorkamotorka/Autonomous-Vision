# main program for all classes
from logger import Logger
import sys
sys.path.append('Extract_Feature')
from qbooster import BoostedFPS
from feats_extra import FeatureExtract
sys.path.append('GUI')
#from pangopango import Pango3D
#from dyconfigure import DynamicConfigure
import cv2 as cv
import numpy as np
from multiprocessing import Process, Queue
import pangolin as pango
import OpenGL.GL as gl


cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.namedWindow('filtered', cv.WINDOW_NORMAL)


def nothing():
    pass

class DynamicConfigure(object):
    def __init__(self, window):
        self.win = window
        cv.createTrackbar('NumKeyPoints', self.win, 100, 1000, nothing)
        #cv.createTrackbar('NumKeyPoints', self.win, 100, 1000, nothing)
        #cv.createTrackbar('NumKeyPoints', self.win, 100, 1000, nothing)
        switch = '0 : OFF \n1 : ON'
        cv.createTrackbar(switch, self.win, 0,1, nothing)
        Thread(target=self.track, args=(), daemon=True).start() 



    def track(self, img):
        num = cv.getTrackbarPos('NumKeyPoints', self.win)
        toggle = cv.getTrackbarPos(switch, self.win)

        if toggle == 0:
            img[:] = 0
        else:
            img[:] = [num]
        
        return img


class Pango3D(object):
    def __init__(self):
        self.state = None
        self.model = None
        self.disp = None
        self.points = []
        self.frame = []
        self.Q = Queue()
        Process(target=self.process_disp, args=(self.Q,), daemon=True).start()

    def process_disp(self, q):
        self.init_disp()
        while True:
                self.update_disp(q)

    def init_disp(self):
        pango.CreateWindowAndBind('Pango3D', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.model = pango.OpenGlRenderState(
                pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
                pango.ModelViewLookAt(1, 1, -2, 0, 0, 0, pango.AxisDirection.AxisY))
        self.handler = pango.Handler3D(self.model)
        self.disp = pango.CreateDisplay()
        self.disp.SetBounds(0.0, 1.0, 0.0, 1.0, 640.0/480.0)
        self.disp.SetHandler(self.handler)

    def update_disp(self, q):
        if self.state is None or not q.empty():
                self.state = q.get()
        
        ppts = np.array(self.state)
        print(ppts.shape)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.disp.Activate(self.model)
        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        pango.DrawPoints(ppts)
                
        pango.FinishFrame()

    def read_pcl(self, keypoints):
        # need to first convert it to proper 3D point! 
        pts = []
        for kps in keypoints:
                pts.append(kps)
        #self.state = pts
        #print(pts)
        self.Q.put(pts) # np.array prev
        #print(self.Q.empty())

    # GET CAMERA POSE ACCORDING TO IMAGES


    #class Point(object):
    #	def __init__(self):
    #		pass

def HomogenousCoord(pts):
    # [x,y] -> [x,y,1]
    pts = np.append(pts, np.ones([pts.shape[0], 1]), axis=1)# commented numpy array in feats_extract
    return pts


def process_frame(img, kpoints):
    cv.imshow('image', img)
    img = cv.drawKeypoints(img, keypoints=kpoints, outImage=None, color=(255,0,0))
    cv.imshow('filtered', img)
    cv.waitKey(1)


pango3d = Pango3D()
log = Logger('slam')

if __name__ == '__main__':
    feats = FeatureExtract()
    booster = BoostedFPS('test_countryroad.mp4')
    #pango3d = Pango3D()
    start = cv.getTickCount()	

    running = True
    while running:
        frame = booster.getFrame()
        _kp, _des, _matches = feats.detectCombo(frame)
        process_frame(frame, _kp)
        #_kp = HomogenousCoord(_kp)
        running = booster.checkBuffer()

    end = cv.getTickCount()		
    exec_time = (end - start)/ cv.getTickFrequency()
    log.info(f"Execution time: {exec_time} seconds")
    booster.stopStream()
    cv.destroyAllWindows()
