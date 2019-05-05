import os
import time
import numpy as np

from Box2D import (b2World, b2Vec2)

from ..gen_world import new_confined_clustered_circles_world
from .run_world import run_world

# ----- Parameters -----
# Number of worlds to use
nWorlds = 10

# Number of bodies in worlds
nBodies = 100
# Something about spread of bodies?
sigma_coef = 1.2
# Dimension of static box
xlow, xhi = 0, 15
ylow, yhi = 0, 15
# body radius min and max
radius = (0.5, 0.5)
# Seeds to use for body generator - NOTE: number of seeds should match number of worlds
seeds = range(12345, 12345+nWorlds)

# Timestep
timeStep = 1.0 / 100
# Iteration limits
velocityIterations = 1000
positionIterations = 250
# Iteration thresholds
velocityThreshold = 1*10**-4
positionThreshold = 1*10**-4
# Number of steps
steps = 250


# Choose a model
#model = NoWarmStartModel(); filename = "none.npz"
#model = BuiltinWarmStartModel(); filename = "builtin.npz"
#model = BadModel(); filename = "bad.npz"
#model = RandomModel(0); filename = "random.npz"
#model = CopyWorldModel(); filename = "copy.npz"
#model = CopyWorldModel(0); filename = "copy0.npz"
#model = CopyWorldModel(2); filename = "copy2.npz"
#model = IdentityGridModel((xlow, ylow), (xhi, yhi), 0.25, 0.25, 0.25); filename = "grid_025.npz"
model = CNNModel(Peak()); filename = "peak.npz"
#model = CNNModel(Pressure()); filename = "pressure.npz"


# File path and name
filepath = "./results/" + filename

# Print various iteration numbers as simulation is running
quiet = True
# Show visualization of world as simulation is running
# note: significantly slower
visualize = False



# ----- Run Simulations -----
results = []
for i in range(nWorlds):
    print("Running world %2d of %2d" % (i+1, nWorlds))
    start = time.time()

    # Create world
    world = b2World()

    # Fill world with static box and circles
    seed = seeds[i]
    new_confined_clustered_circles_world(
        world,
        nBodies,
        b2Vec2(xlow, ylow),
        b2Vec2(xhi, yhi),
        radius,
        sigma_coef,
        seed,
    )

    # Run
    result = run_world(
        world,
        timeStep,
        steps,
        velocityIterations,
        positionIterations,
        velocityThreshold = velocityThreshold,
        positionThreshold = positionThreshold,
        model = model,
        iterations = True,
        convergenceRates = True,
        lambdaErrors = True,
        quiet = quiet,
        visualize = visualize,
    )
    results.append(result)

    print("Simulation took: %d s" % (time.time()-start))



# ----- Save Results -----
print("Saving Results")

# If path is not absolute, fix it
if not os.path.isabs(filepath):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(file_dir, filepath)

np.savez(
    filepath,
    results = results,
)
