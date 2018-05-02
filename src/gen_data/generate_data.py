import time
import cv2
import numpy as np

from Box2D import (b2World, b2Vec2, b2ContactListener)

from ..gen_world import new_confined_clustered_circles_world
from ..sim_types import SimData
from ..xml_writing.b2d_2_xml import XMLExporter
from ..opencv_draw import OpencvDrawFuncs


# ----- Parameters -----
# Number of bodies in world
nBodies = 100
# Seeds to use for body generator - determines the number of datasets created
seeds = range(1)
# Something about spread of bodies?
sigma_coef = 1.2
# Dimension of static box
xlow, xhi = 10, 60
ylow, yhi = 10, 60
# body radius min and max
r = (1, 1)

# Timestep
timeStep = 1.0 / 100
# Iteration limits
velocityIterations = 5000
positionIterations = 2500
# Iteration thresholds
velocityThreshold = 6*10**-5
positionThreshold = 2*10**-5
# Number of steps
steps = 1000

# Path to directory where data should be stored, relative to the xml_writing directory
path = "../gen_data/data/"
# Decides whether to store configurations without any contacts
skipContactless = True

# Print various iteration numbers as simulation is running
quiet = False
# Show visualization of world as simulation is running
# note: significantly slower
visualize = True


# NOTE: Multiple contacts between objects is currently not supported,
# meaning that bodies in the corners of the static box, i.e. with two
# separate contact points with the box, will not be stored correctly.


# ----- Misc -----
# B2ContactListener for recording impulses
class ImpulseListener(b2ContactListener):
    def __init__(self):
        super(ImpulseListener, self).__init__()
        self.__reset__()

    def __reset__(self):
        self.impulses = {}

    def PostSolve(self, contact, impulse):
        pc = contact.manifold.pointCount
        for i in range(pc):
            master = contact.fixtureA.body.userData.id
            slave  = contact.fixtureB.body.userData.id

            normal  = impulse.normalImpulses[i]
            tangent = impulse.tangentImpulses[i]

            self.impulses[(master, slave)] = (normal, tangent)


# Define a drawer if set
if visualize:
    drawer = OpencvDrawFuncs(w=640, h=640, ppm=10)
    drawer.install()


# ----- Data generation -----
listener = ImpulseListener()
for i in range(len(seeds)):
    seed = seeds[i]
    print("Running world %d of %d" % (i+1, len(seeds)))

    # Create world
    world = b2World()
    world.userData = SimData(name=str(seed), d_t=timeStep)
    world.contactListener = listener

    # Fill world with static box and circles
    new_confined_clustered_circles_world(world, nBodies, b2Vec2(xlow, ylow), b2Vec2(xhi, yhi), r, sigma_coef, seed)

    # Set iteration thresholds
    world.velocityThreshold = velocityThreshold
    world.positionThreshold = positionThreshold

    # Initialize XML exporter
    exp = XMLExporter(world, path)

    # Run simulation
    for i in range(steps):
        if not quiet:
            print("step", i)

        # Reset the impulse dict
        listener.__reset__()

        # Save snapshot of current configuration
        exp.__reset__()
        exp.snapshot_bodies()
        exp.snapshot_contacts()

        # Tell the world to take a step
        world.Step(timeStep, velocityIterations, positionIterations)
        world.userData.tick()
        world.ClearForces()

        # Draw the world
        if visualize:
            drawer.clear_screen()
            drawer.draw_world(world)

            cv2.imshow('World', drawer.screen)
            cv2.waitKey(25)

        # Store contact impulses
        exp.snapshot_impulses(listener.impulses)
        #print(listener.impulses)

        # Save the snapshot if wanted
        if not skipContactless or world.GetProfile().contactsSolved > 0:
            exp.save_snapshot()
