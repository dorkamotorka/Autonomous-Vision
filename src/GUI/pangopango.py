import numpy as np
from multiprocessing import Process, Queue
import pangolin as pango
import OpenGL.GL as gl
import cv2 as cv

class Pango3D(object):
	def __init__(self):
		self.state = None
		self.model = None
		self.disp = None
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
		self.state = None
		while not q.empty():
			self.state = q.get()
		#while not pango.ShouldQuit():
		#print(self.state)
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		self.disp.Activate(self.model)
		gl.glPointSize(5)
		gl.glColor3f(1.0, 0.0, 0.0)
		if self.state is not None:
			pango.DrawPoints(self.state)
			
		pango.FinishFrame()

	def read_pcl(self, keypoints):
		#print(keypoints) # get points but not in Q because of numpy array problem with multiprocessing
		# not working because data not pickled
		pts = []
		for kps in keypoints:
			pts.append(kps)
		self.Q.put(np.array(pts))
		print(self.Q.empty())
