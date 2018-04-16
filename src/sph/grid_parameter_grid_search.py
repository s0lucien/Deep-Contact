import numpy as np

from itertools import product
from Box2D import (b2World, b2Vec2, b2_dynamicBody)

from gen_world import new_confined_clustered_circles_world

from .gridsplat import SPHGridManager

# Number of worlds to generate
nWorlds = 25
# Number of bodies in world
nBodies = 100
# Something about spread of bodies?
sigma_coef = 1.2
# Dimension of static box
p_ll = (-30, 0)
p_ur = (30, 60)
# Radius of bodies
r = (1, 1)

# Create world
worlds = [b2World() for _ in range(nWorlds)]
# Fill worlds with static box and circles
for i in range(nWorlds):
    new_confined_clustered_circles_world(worlds[i], nBodies, b2Vec2(p_ll), b2Vec2(p_ur), r, sigma_coef, i)

# Determine original totals - change attribute here if wanted
originals = []
for w in worlds:
    original = 0
    for b in w.bodies:
        original += b.mass
    originals.append(original)
# Attribute to create grids for
attribute = "mass"


# Grid parameters to try
# Grid resolution
res = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]
# Support radius
h = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]

# Create all combinations of grid parameters
parameters = list(product(*[res, h]))
nParameters = len(parameters)

# Create grid managers, tell them to create grids, and query for all bodies in world
averages = []
for i in range(nParameters):
    p = parameters[i]

    print("Trying parameter set %d of %d" % (i, nParameters))

    differences = []
    for j in range(nWorlds):
        world = worlds[j]

        # Create grid and step
        gm = SPHGridManager(world, p_ll, p_ur, p[0], p[0], p[1])
        gm.Step([attribute])

        # Query for all bodies and sum
        total = 0
        for b in world.bodies:
            if b.type is b2_dynamicBody:
                total += gm.query(b.position.x, b.position.y, attribute)

        # Store results
        differences.append(abs(originals[j] - total))

    # Averate results and store
    averages.append(np.mean(differences))


# Combine averages with parameters
pairs = list(zip(averages, parameters))

# Sort pairs
sortedPairs = sorted(pairs, key=lambda p: p[0])

# Print results
print("{0:20s}     ({1:3s}, {2:3s})".format("Absolute differences", "res", "h"))
for d, p in sortedPairs:
    print("{0:20f}     ({1:1.1f}, {2:1.1f})".format(d, p[0], p[1]))
