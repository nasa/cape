#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'VisItTecplotBinaryReader'
arrow_tec_boundary_timestep200plt = VisItTecplotBinaryReader(FileName=['arrow_tec_boundary.plt'])
arrow_tec_boundary_timestep200plt.MeshStatus = ['zone']
arrow_tec_boundary_timestep200plt.PointArrayStatus = []

# Properties modified on arrow_tec_boundary_timestep200plt
arrow_tec_boundary_timestep200plt.PointArrayStatus = ['x', 'y', 'z', 'cp']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
renderView1.ViewSize = [1033, 810]

# show data in view
arrow_tec_boundary_timestep200pltDisplay = Show(arrow_tec_boundary_timestep200plt, renderView1)
# trace defaults for the display properties.
arrow_tec_boundary_timestep200pltDisplay.ColorArrayName = [None, '']
arrow_tec_boundary_timestep200pltDisplay.GlyphType = 'Arrow'
arrow_tec_boundary_timestep200pltDisplay.ScalarOpacityUnitDistance = 0.4701380898762916

# reset view to fit data
renderView1.ResetCamera()

# set scalar coloring
ColorBy(arrow_tec_boundary_timestep200pltDisplay, ('FIELD', 'mach'))

# show color bar/color legend
arrow_tec_boundary_timestep200pltDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'vtkBlockColors'
vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
vtkBlockColorsLUT.InterpretValuesAsCategories = 1
vtkBlockColorsLUT.Annotations = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '8', '9', '9', '10', '10', '11', '11']
vtkBlockColorsLUT.ActiveAnnotatedValues = ['0', '1', '2']
vtkBlockColorsLUT.IndexedColors = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.63, 0.63, 1.0, 0.67, 0.5, 0.33, 1.0, 0.5, 0.75, 0.53, 0.35, 0.7, 1.0, 0.75, 0.5]

# get opacity transfer function/opacity map for 'vtkBlockColors'
vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')

# set scalar coloring
ColorBy(arrow_tec_boundary_timestep200pltDisplay, ('POINTS', 'cp'))

# rescale color and/or opacity maps used to include current data range
arrow_tec_boundary_timestep200pltDisplay.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
arrow_tec_boundary_timestep200pltDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'cp'
cpLUT = GetColorTransferFunction('cp')
cpLUT.RGBPoints = [
    -0.5, 0.231373, 0.298039, 0.752941,
    0.00, 0.865003, 0.865003, 0.865003, 
    1.25, 0.705882, 0.015686, 0.149020]
cpLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'cp'
cpPWF = GetOpacityTransferFunction('cp')
cpPWF.Points = [-0.5, 0.0, 0.5, 0.0, 1.25, 1.0, 0.5, 0.0]
cpPWF.ScalarRangeInitialized = 1

# create a new 'VisItTecplotBinaryReader'
arrow_planey0_timestep200plt = VisItTecplotBinaryReader(FileName=['arrow_plane-y0.plt'])
arrow_planey0_timestep200plt.MeshStatus = ['plane_y0']
arrow_planey0_timestep200plt.PointArrayStatus = []

# Properties modified on arrow_planey0_timestep200plt
arrow_planey0_timestep200plt.PointArrayStatus = ['x', 'y', 'z', 'mach']

# show data in view
arrow_planey0_timestep200pltDisplay = Show(arrow_planey0_timestep200plt, renderView1)
# trace defaults for the display properties.
arrow_planey0_timestep200pltDisplay.ColorArrayName = [None, '']
arrow_planey0_timestep200pltDisplay.GlyphType = 'Arrow'
arrow_planey0_timestep200pltDisplay.ScalarOpacityUnitDistance = 14.817648284832933

# set scalar coloring
ColorBy(arrow_planey0_timestep200pltDisplay, ('POINTS', 'mach'))

# rescale color and/or opacity maps used to include current data range
arrow_planey0_timestep200pltDisplay.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
arrow_planey0_timestep200pltDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'mach'
machLUT = GetColorTransferFunction('mach')
machLUT.RGBPoints = [0.04105953127145767, 0.29411764705882354, 0.01568627450980392, 0.6352941176470588, 1.007740754634142, 1.0, 1.0, 1.0, 1.9744219779968262, 0.0, 0.6666666666666666, 0.0]
machLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'mach'
machPWF = GetOpacityTransferFunction('mach')
machPWF.Points = [0.04105953127145767, 0.0, 0.5, 0.0, 1.9744219779968262, 1.0, 0.5, 0.0]
machPWF.ScalarRangeInitialized = 1

# Properties modified on machLUT
machLUT.RGBPoints = [
    0.0, 0.24313725490196078, 0.0196078431372549, 0.6352941176470588, 
    1.0, 1.0, 1.0, 1.0, 
    2.0, 0.0, 0.6039215686274509, 0.0]

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [-0.7464934106579985, -19.002488492507666, -1.4317972232017544]
renderView1.CameraFocalPoint = [3.7121746084643736, -3.266085056404947, -0.3569719255235101]
renderView1.CameraViewUp = [0.10773909097186195, -0.09809910151154101, 0.9893274758941982]
renderView1.CameraParallelScale = 4.2423383544520785

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).

# save screenshot
SaveScreenshot('slice-y0.png', magnification=1, quality=100, view=renderView1)

#### saving camera placements for all active views
