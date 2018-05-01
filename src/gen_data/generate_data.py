import time
import numpy as np

from Box2D import (b2World, b2Vec2)

from ..gen_world import new_confined_clustered_circles_world
from ..warm_starting.warm_start import run_world
from ..sim_types import SimData

# ----- Parameters -----
# Number of bodies in world
nBodies = 100
# Seeds to use for body generator - determines the number of datasets created
seeds = range(2)
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

# Print various iteration numbers as simulation is running
printing = False
# Show visualization of world as simulation is running
# note: significantly slower
visualize = False

# Data generation
for i in range(len(seeds)):
    seed = seeds[i]
    print("Running world %d of %d" % (i+1, len(seeds)))

    # Create world
    world = b2World()
    world.userData = SimData(name=str(seed), d_t=timeStep)

    # Fill world with static box and circles
    new_confined_clustered_circles_world(world, nBodies, b2Vec2(xlow, ylow), b2Vec2(xhi, yhi), r, sigma_coef, seed)

    # Run simulation
    run_world(world, timeStep, steps,
              velocityIterations, positionIterations,
              velocityThreshold=velocityThreshold, positionThreshold=positionThreshold,
              storeAsXML=True, path=path,
              quiet=not printing, visualize=visualize)
