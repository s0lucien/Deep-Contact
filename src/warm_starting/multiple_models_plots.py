import time
import numpy as np
import matplotlib.pyplot as plt

from .no_warmstart_model import NoWarmStartModel
from .builtin_warmstart_model import BuiltinWarmStartModel
from .bad_model import BadModel
from .random_model import RandomModel
from .parallel_world_model import ParallelWorldModel
from .copy_world_model import CopyWorldModel
from .identity_grid_model import IdentityGridModel

from Box2D import (b2World, b2Vec2)

from gen_world import new_confined_clustered_circles_world
from .warm_start import run_world


# ----- Parameters -----
# Number of bodies in world
nBodies = 100
# Seed to use for body generator
seed = 101
# Something about spread of bodies?
sigma_coef = 1.2
# Dimension of static box
xlow, xhi = -30, 30
ylow, yhi = 0, 60

# Timestep
timeStep = 1.0 / 100
# Iteration limits
velocityIterations = 20000
positionIterations = 10000
# Iteration thresholds
velocityThreshold = 10**-4
positionThreshold = 10**-5
# Number of steps
steps = 1000

# Grid parameters - only relevant for identity grid model
# Grid lower left point
p_ll = (xlow, ylow)
# Grid upper right point
p_ur = (xhi, yhi)
# Grid x-resolution
xRes = 0.5
# Grid y-resolution
yRes = 0.5
# Support radius
h = 3

# Number of models to use
nModels = 3
# Create worlds in case models needs it
worlds = []
for _ in range(nModels):
    world = b2World()
    # Fill world with static box and circles
    new_confined_clustered_circles_world(world, nBodies, b2Vec2(p_ll), b2Vec2(p_ur), (1, 1), sigma_coef, seed)
    worlds.append(world)
# Choose the models
models = [
    NoWarmStartModel(),
    BuiltinWarmStartModel(),
    IdentityGridModel(worlds[2], p_ll, p_ur, xRes, yRes, h)
]
# Model names for plots
names = [
    "None",
    "Builtin",
    "Grid"
]

# Convergence rate plots
plotConvergenceRates = True
# Choose convergence rate data
# 1 for velocity lambda two norms
# 2 for velocity lambda infinity norms
# 3 for position lambdas
convergenceRateData = 1
# Cutoff for convergence rate plot
cutoff = 500



# ----- Run simulation -----
results = []
for i in range(nModels):
    print("Running simulation %d of %d" % (i+1, nModels))
    results.append(run_world(worlds[i], models[i], timeStep, steps,
                             velocityIterations, positionIterations, velocityThreshold, positionThreshold,
                             iterations=False, convergenceRates=plotConvergenceRates, quiet=True))

if plotConvergenceRates:
    twentyfivePercentages  = []
    fiftyPercentages       = []
    seventyfivePercentages = []
    for i in range(nModels):
        if convergenceRateData == 1:
            convergenceData = results[i]["velocityLambdaTwoNorms"]
        if convergenceRateData == 2:
            convergenceData = results[i]["velocityLambdaInfNorms"]
        if convergenceRateData == 3:
            convergenceData = results[i]["positionLambdas"]

        # Pad convergence rates to be same length using NaNs
        pad = len(max(convergenceData, key=len))
        convergenceRatesArray = np.array([i + [np.NaN]*(pad-len(i)) for i in convergenceData])

        #tenPercent         = np.nanpercentile(convergenceRatesArray, 10, axis=0)
        twentyfivePercentages.append(np.nanpercentile(convergenceRatesArray, 25, axis=0))
        fiftyPercentages.append(np.nanpercentile(convergenceRatesArray, 50, axis=0))
        seventyfivePercentages.append(np.nanpercentile(convergenceRatesArray, 75, axis=0))
        #ninetyPercent      = np.nanpercentile(convergenceRatesArray, 90, axis=0)



# ----- Plot stuff -----
# Make an overall plot title with a few stats
def pretty(s):
    return '{0:.0E}'.format(s)

titleStats = "nBodies = " + str(nBodies) + ", dt = " + pretty(timeStep) + "\n"
titleStats += "vel_iter = " + pretty(velocityIterations)
titleStats += ", vel_thres = " + pretty(world.velocityThreshold) + "\n"
titleStats += "pos_iter = " + pretty(positionIterations)
titleStats += ", pos_thres = " + pretty(world.positionThreshold)


# Convergence rate plots
if plotConvergenceRates:
    if convergenceRateData == 1:
        title = "Velocity Lambda Two-Norms"
    if convergenceRateData == 2:
        title = "Velocity Lambda Inf-Norms"
    if convergenceRateData == 3:
        title = "Position Lambdas"

    fig = plt.figure()
    fig.suptitle(title + "\n" + titleStats)

    # Full convergence rates
    ax1 = fig.add_subplot(221)
    for f in fiftyPercentages:
        ax1.semilogy(f)
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(title)
    ax1.set_title("Convergence Rate - All iterations, 50")

    # Limited convergence rates
    ax1 = fig.add_subplot(222)
    for f in fiftyPercentages:
        ax1.semilogy(f)
    ax1.set_xlim([0, cutoff])
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(title)
    ax1.set_title("Convergence Rate - Cufoff, 50")

    # Limited convergence rates - 25
    ax1 = fig.add_subplot(223)
    for f in twentyfivePercentages:
        ax1.semilogy(f)
    ax1.set_xlim([0, cutoff])
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(title)
    ax1.set_title("Convergence Rate - Cufoff, 25")

    # Limited convergence rates - 75
    ax1 = fig.add_subplot(224)
    for f in seventyfivePercentages:
        ax1.semilogy(f)
    ax1.set_xlim([0, cutoff])
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(title)
    ax1.set_title("Convergence Rate - Cufoff, 75")


if plotConvergenceRates:
    plt.show()
