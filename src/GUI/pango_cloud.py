import pangolin as pango
import OpenGL.GL as gl
import numpy as np

pango.CreateWindowAndBind('PangoPointCloud', 640, 480)
gl.glEnable(gl.GL_DEPTH_TEST)

# Define Projection and initial ModelView matrix
scam = pango.OpenGlRenderState(
    pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
    pango.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pango.AxisDirection.AxisY))
handler = pango.Handler3D(scam)

# Create Interactive View in window
dcam = pango.CreateDisplay()
dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
dcam.SetHandler(handler)

while not pango.ShouldQuit():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    dcam.Activate(scam)
    
    # Render OpenGL Cube
    pango.glDrawColouredCube()

    # Draw Point Cloud
    points = np.random.random((100000, 3)) * 10
    gl.glPointSize(2)
    gl.glColor3f(1.0, 0.0, 0.0)
    pango.DrawPoints(points)

    pango.FinishFrame()
