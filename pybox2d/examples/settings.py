

class fwSettings(object):
    # The default backend to use in (can be: pyglet, pygame, etc.)
    backend = 'pygame'

    # Physics options
    timeStep = 1e-3
    hz=1/timeStep
    velocityIterations = 8
    positionIterations = 3
    velocityThreshold = 10**-4
    positionThreshold = 10**-5
    # Makes physics results more accurate (see Box2D wiki)
    enableWarmStarting = True
    enableContinuous = True     # Calculate time of impact
    enableSubStepping = False

    # Drawing
    drawShapes = True
    drawJoints = True
    drawCoreShapes = False
    drawAABBs = False
    drawOBBs = False
    drawPairs = True
    drawContactPoints = False
    maxContactPoints = 100
    drawContactNormals = False
    drawCOMs = False            # Centers of mass
    pointSize = 2.5             # pixel radius for drawing points

    # Miscellaneous testbed options
    pause = False
    singleStep = False
    # run the test's initialization without graphics, and then quit (for
    # testing)
    onlyInit = False



from optparse import OptionParser

parser = OptionParser()
list_options = [i for i in dir(fwSettings)
                if not i.startswith('_')]

for opt_name in list_options:
    value = getattr(fwSettings, opt_name)
    if isinstance(value, bool):
        if value:
            parser.add_option('', '--no-' + opt_name, dest=opt_name,
                              default=value, action='store_' + str(not value).lower(),
                              help="don't " + opt_name)
        else:
            parser.add_option('', '--' + opt_name, dest=opt_name, default=value,
                              action='store_' + str(not value).lower(),
                              help=opt_name)

    else:
        if isinstance(value, int):
            opttype = 'int'
        elif isinstance(value, float):
            opttype = 'float'
        else:
            opttype = 'string'
        parser.add_option('', '--' + opt_name, dest=opt_name, default=value,
                          type=opttype,
                          help='sets the %s option' % (opt_name,))


fwSettings, args = parser.parse_args()
