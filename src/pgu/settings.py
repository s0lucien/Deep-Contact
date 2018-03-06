
#             text                  variable
checkboxes = (("Warm Starting", "enableWarmStarting"),
              ("Time of Impact", "enableContinuous"),
              ("Sub-Stepping", "enableSubStepping"),
              ("Draw", None),
              ("Shapes", "drawShapes"),
              ("Joints", "drawJoints"),
              ("AABBs", "drawAABBs"),
              ("Pairs", "drawPairs"),
              ("Contact Points", "drawContactPoints"),
              ("Contact Normals", "drawContactNormals"),
              ("Center of Masses", "drawCOMs"),
              ("Statistics", "drawStats"),
              ("FPS", "drawFPS"),
              ("Control", None),
              ("Pause", "pause"),
              ("Single Step", "singleStep"))

sliders = [
    {'name': 'hz', 'text': 'Hertz', 'min': 5, 'max': 200},
    {'name': 'positionIterations', 'text': 'Pos Iters', 'min': 0, 'max': 100},
    {'name': 'velocityIterations', 'text': 'Vel Iters', 'min': 1, 'max': 500},
]

class uiSettings(object):
    backend = 'pygame'

    # Drawing
    drawStats = False
    drawShapes = True
    drawJoints = False
    drawCoreShapes = False
    drawAABBs = False
    drawOBBs = False
    drawPairs = False
    drawContactPoints = False
    maxContactPoints = 100
    drawContactNormals = False
    drawFPS = False
    drawMenu = False             # toggle by pressing F1
    drawCOMs = False            # Centers of mass
    pointSize = 2.5             # pixel radius for drawing points

    # Miscellaneous testbed options
    pause = False
    singleStep = False