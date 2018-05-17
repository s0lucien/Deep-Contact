import numpy as np

from itertools import product
from Box2D import (b2World, b2Vec2, b2_dynamicBody)

from ..gen_world import new_confined_clustered_circles_world

from .gridsplat import SPHGridManager, world_body_dataframe, world_contact_dataframe

# Number of worlds to generate
nWorlds = 25
# Number of bodies in world
nBodies = 50
# Something about spread of bodies?
sigma_coef = 1.2
# Dimension of static box
p_ll = (0, 0)
p_ur = (50, 50)
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
worldTotalDifferenceAverages = [.0]*nParameters
bodyDifferenceAverageAverages = [.0]*nParameters
for i in range(nParameters):
    p = parameters[i]

    print("Trying parameter set %d of %d" % (i, nParameters))

    worldTotalDifferences = [.0]*nWorlds
    bodyDifferenceAverages = [.0]*nWorlds
    for j in range(nWorlds):
        world = worlds[j]

        # Create data frames
        df_b = world_body_dataframe(world)

        # Create gridmanager
        gm = SPHGridManager(p_ll, p_ur, p[0], p[0], p[1])

        # Tell it to create grids and interp functions
        grid = gm.create_grids(df_b, channels=[attribute])
        #gm.create_interp([attribute])
        gm.create_tree([attribute])

        # Query for all bodies and sum
        worldTotal = 0
        bodyDifferences = [.0]*nBodies
        k = 0
        for b in world.bodies:
            if b.type is b2_dynamicBody:
                #value = gm.query_interp(b.position.x, b.position.y, attribute)
                value = gm.query_tree([b.position.x], [b.position.y], attribute)
                worldTotal += value
                bodyDifferences[k] = abs(b.mass - value)
                k += 1

        # Store results
        worldTotalDifferences[j] = abs(originals[j] - worldTotal)
        bodyDifferenceAverages[j] = np.mean(bodyDifferences)

    # Averate results and store
    worldTotalDifferenceAverages[i] = np.mean(worldTotalDifferences)
    bodyDifferenceAverageAverages[i] = np.mean(bodyDifferenceAverages)

# Combine averages with parameters
pairs = list(zip(parameters, worldTotalDifferenceAverages, bodyDifferenceAverageAverages))

# Sort pairs
sortedPairs = sorted(pairs, key=lambda p: p[1])

# Print results
print("Original total: {0:3.2f}".format(originals[0]))
print("Body mass:      {0:3.2f}".format(worlds[0].bodies[1].mass))
print("({0:4s}, {1:4s})\t{2:14s}\t{3:13s}".format("res", "h", "Avg total diff", "Avg body diff"))
for p, t, b in sortedPairs:
    print("({0:1.2f}, {1:1.2f})\t{2:11.2f}\t{3:10.2f}".format(p[0], p[1], t, b))
