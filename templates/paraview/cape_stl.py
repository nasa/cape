try: paraview.simple
except: from paraview.simple import *

# Command-line processing
import sys
# Need a cross product
from numpy import cross

# Process an input vector
def read_vector(o, vdef):
    # Check type and contents for shortcuts
    if o in ['x']:
        v = [1.0, 0.0, 0.0]
    elif o in ['-x']:
        v = [-1.0, 0.0, 0.0]
    elif o in ['y']:
        v = [0.0, 1.0, 0.0]
    elif o in ['-y']:
        v = [0.0, -1.0, 0.0]
    elif o in ['z']:
        v = [0.0, 0.0, 1.0]
    elif o in ['-z']:
        v = [0.0, 0.0, -1.0]
    elif type(o).__name__=="str" and o.startswith('['):
        v = eval(o)
        if len(v) != 3:
            raise IOError('Axis definition "%s" is not a valid vector' % o)
    else:
        v = vdef
    return v

# Process the arguments
if len(sys.argv) < 2:
    o_r = 'x'
    o_u = 'y'
elif len(sys.argv) == 2:
    o_r = sys.argv[1]
    o_u = 'y'
else:
    o_r = sys.argv[1]
    o_u = sys.argv[2]
# Read the vector that points to the right.
v_r = read_vector(o_r, [1.0, 0.0, 0.0])
# Read the vector that points up
v_u = read_vector(o_u, [0.0, 1.0, 0.0])
# Process inward (focus) vector
v_f = cross(v_u, v_r)
# Check for parallel vectors)
if sqrt(v_f[0]**2 + v_f[1]**2 + v_f[2]**2) <= 1e-8:
    raise IOError("Right and upward vectors are parallel")

# Read the STL file
Components_i_stl = STLReader( FileNames=['comp.stl'] )

# Open a "view"
RenderView1 = GetRenderView()
DataRepresentation2 = Show()
DataRepresentation2.EdgeColor = [0.0, 0.0, 0.0]
DataRepresentation2.Representation = 'Surface With Edges'

# Set camera position
RenderView1.CameraPosition = [0.0, 0.0, 0.0]
RenderView1.CameraFocalPoint = v_f
RenderView1.CameraViewUp = v_u
RenderView1.ResetCamera()

# Make axes have the appropriate settings
RenderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
RenderView1.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
RenderView1.CenterAxesVisibility = 0
# Background color
RenderView1.Background = [1.0, 1.0, 1.0]

# Show the image
Render()
# Save the image
WriteImage('cape_stl.png', Magnification=2)

