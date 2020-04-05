import numpy as np
from threading import Thread
from queue import Queue
import pangolin as pango
import OpenGL.GL as gl
import cv2 as cv

class Pango3D(object):
	def __init__(self, winTitle='Pango3D', width=640, height=480): # fix
		self.winTitle = winTitle 
		self.width = width
		self.height = height
		self.model = None
		self.disp = None
		self.points = []
		self.Q = Queue()
		Thread(target=self.process_disp).start()

	def process_disp(self):
		self.init_disp(self.winTitle, self.width, self.height)
		while not pango.ShouldQuit():
			#print('inside')
			if self.Q.empty():
				continue
			gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
			self.disp.Activate(self.model)
			#pango.glDrawColouredCube()
			# Draw Point Cloud
			#self.points = np.random.random((100000, 3)) 
			gl.glPointSize(6)
			gl.glColor3f(1.0, 0.0, 0.0)
			self.points = self.Q.get()
			print(self.points.shape)
			pango.DrawPoints(self.points)
			pango.FinishFrame()

	def init_disp(self, win, w, h):
		pango.CreateWindowAndBind(win, w, h)
		gl.glEnable(gl.GL_DEPTH_TEST)
		self.model = pango.OpenGlRenderState(
			pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
			pango.ModelViewLookAt(1, 1, -2, 0, 0, 0, pango.AxisDirection.AxisY))
		handler = pango.Handler3D(self.model)
		self.disp = pango.CreateDisplay()
		self.disp.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
		self.disp.SetHandler(handler)

	def read_pcl(self, keypoints):
		print(keypoints.shape)
		self.Q.put(keypoints)
