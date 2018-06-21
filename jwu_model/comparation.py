import time
import numpy as np
import matplotlib

# fix some bugs in darwin
import tkinter as tk
root = tk.Tk()
import matplotlib.pyplot as plt

from src.warm_starting.builtin_warmstart_model import BuiltinWarmStartModel
from src.warm_starting.bad_model import BadModel
from src.warm_starting.random_model import RandomModel
from src.warm_starting.parallel_world_model import ParallelWorldModel
from src.warm_starting.copy_world_model import CopyWorldModel
from src.warm_starting.identity_grid_model import IdentityGridModel
from src.warm_starting.warm_start import run_world
from cnn_training import loss_func, mean_absolute_loss

from jwu_model.cnn_grid_model import CnnIdentityGridModel

from Box2D import b2World, b2Vec2

from src.gen_world import new_confined_clustered_circles_world

# ----- Parameters -----
# Number of bodies in world
nBodies = 100
# Seed to use for body generator
seed = 123
# Something about spread of bodies?
sigma_coef = 1.2
# Dimension of static box
xlow, xhi = 0, 30
ylow, yhi = 0, 30
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
steps = 600


# Number of models to use
nModels = 4
# Create worlds in case models needs them
worlds = []
for _ in range(nModels):
    world = b2World()
    # Fill world with static box and circles
    new_confined_clustered_circles_world(world, nBodies, b2Vec2(xlow, ylow), b2Vec2(xhi, yhi), r, sigma_coef, seed)
    worlds.append(world)

# Choose the models, each as a pair of a model and a name.
# In case a model needs inputs, make sure to provide them, and in particular make sure to use the correct world
from keras import models
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-p', '--model_path', dest='path')

options, _ = parser.parse_args()

solver = models.load_model(
    options.path,
    custom_objects={
        'loss_func': loss_func,
        'mean_absolute_loss': mean_absolute_loss,
    }
)
p_ll = (xlow, ylow)
# Grid upper right point
p_ur = (xhi, yhi)
# Grid x-resolution
xRes = 0.75
# Grid y-resolution
yRes = 0.75
# Support radius
h = 0.5
cnn_model = CnnIdentityGridModel(p_ll, p_ur, xRes, yRes, h, solver)

models = [
    (None, "None"),
    (BuiltinWarmStartModel(), "Builtin"),
    (cnn_model, "CNN"),
    (CopyWorldModel(), "Copy")
]


# Velocity convergence rate plots
plotVelocityConvergenceRates = True
# Percentile to plot
# 0 for min, 25 for 1st quantile, 50 for median, 75 for 3rd quantile, 100 for max
velocityPercentile = 50
# Cutoff for convergence rate plot
velocityCutoff = 1000

# Position convergence rate plots
plotPositionConvergenceRates = False
# Percentile to plot
positionPercentile = 50
# Cutoff for convergence rate plot
positionCutoff = 200

# Print various iteration numbers as simulation is running
printing = False
# Show visualization of world as simulation is running
# note: significantly slower
visualize = False



# ----- Run simulation -----
assert len(models) == nModels

conv = plotVelocityConvergenceRates or plotPositionConvergenceRates
results = []
for i in range(nModels):
    print("Running simulation %d of %d" % (i+1, nModels))
    results.append(run_world(worlds[i], timeStep, steps,
                             velocityIterations, positionIterations,
                             velocityThreshold=velocityThreshold, positionThreshold=positionThreshold,
                             model=models[i][0], iterations=False, convergenceRates=conv,
                             quiet=not printing,visualize=visualize))



# ----- Process data -----
if plotVelocityConvergenceRates:
    velocityTwoData = [[] for _ in range(nModels)]
    velocityInfData = [[] for _ in range(nModels)]
    velocityIteratorCounts = [[] for _ in range(nModels)]
    for i in range(nModels):
        velocityLambdaTwoNorms = results[i]["velocityLambdaTwoNorms"]
        velocityLambdaInfNorms = results[i]["velocityLambdaInfNorms"]

        # Pad convergence rates to be same length using NaNs
        pad = len(max(velocityLambdaTwoNorms, key=len))
        velocityTwoArray = np.array([i + [np.NaN]*(pad-len(i)) for i in velocityLambdaTwoNorms])
        velocityInfArray = np.array([i + [np.NaN]*(pad-len(i)) for i in velocityLambdaInfNorms])

        velocityTwoData[i] = np.nanpercentile(velocityTwoArray, velocityPercentile, axis=0)
        velocityInfData[i] = np.nanpercentile(velocityInfArray, velocityPercentile, axis=0)
        velocityIteratorCounts[i] = results[i]["velocityIteratorCounts"]


if plotPositionConvergenceRates:
    positionData = [[] for _ in range(nModels)]
    positionIteratorCounts = [[] for _ in range(nModels)]
    for i in range(nModels):
        positionLambdas = results[i]["positionLambdas"]

        # Pad convergence rates to be same length using NaNs
        pad = len(max(positionLambdas, key=len))
        positionLambdasArray = np.array([i + [np.NaN]*(pad-len(i)) for i in positionLambdas])

        positionData[i] = np.nanpercentile(positionLambdasArray, positionPercentile, axis=0)
        positionIteratorCounts[i] = results[i]["positionIteratorCounts"]



# ----- Plot stuff -----
# Make an overall plot title with a few stats
def pretty(s):
    return '{0:.0E}'.format(s)

titleStats = "nBodies = " + str(nBodies) + ", dt = " + pretty(timeStep) + ", steps = " + pretty(steps) + "\n"
titleStats += "vel_iter = " + pretty(velocityIterations)
titleStats += ", vel_thres = " + pretty(world.velocityThreshold) + "\n"
titleStats += "pos_iter = " + pretty(positionIterations)
titleStats += ", pos_thres = " + pretty(world.positionThreshold)


# Convergence rate plots
if plotVelocityConvergenceRates:
    names = [p[1] for p in models]

    fig = plt.figure("Velocity Convergence")
    fig.suptitle("Velocity Convergence Rates\n" + titleStats)

    # Full lambda two-norm convergence rates
    ax1 = fig.add_subplot(221)
    for data in velocityTwoData:
        ax1.semilogy(data)
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda Two-Norm")
    ax1.set_title("Velocity Lambda Two-Norm Convergence Rate - All iterations, " + str(velocityPercentile) + "%")

    # Counters
    ax1 = fig.add_subplot(222)
    for count in velocityIteratorCounts:
        ax1.plot(count)
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

    # Limited lambda two-norm convergence rates
    ax1 = fig.add_subplot(223)
    for data in velocityTwoData:
        ax1.semilogy(data)
    ax1.set_xlim([0, velocityCutoff])
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda Two-Norm")
    ax1.set_title("Velocity Lambda Two-Norm Convergence Rate - Cutoff, " + str(velocityPercentile) + "%")

    # Limited lambda inf-norm convergence rates
    ax1 = fig.add_subplot(224)
    for data in velocityInfData:
        ax1.semilogy(data)
    ax1.set_xlim([0, velocityCutoff])
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda Inf-Norm")
    ax1.set_title("Velocity Lambda Inf-Norm Convergence Rate - Cutoff, " + str(velocityPercentile) + "%")


if plotPositionConvergenceRates:
    names = [p[1] for p in models]

    fig = plt.figure("Position Convergence")
    fig.suptitle("Position Convergence Rates\n" + titleStats)

    # Full lambda convergence rates
    ax1 = fig.add_subplot(221)
    for data in positionData:
        ax1.semilogy(data)
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda")
    ax1.set_title("Position Lambda Convergence Rate - All iterations, " + str(positionPercentile) + "%")

    # Counters
    ax1 = fig.add_subplot(222)
    for count in positionIteratorCounts:
        ax1.plot(count)
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

    # Limited lambda convergence rates
    ax1 = fig.add_subplot(223)
    for data in positionData:
        ax1.semilogy(data)
    ax1.set_xlim([0, positionCutoff])
    ax1.legend(names)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda")
    ax1.set_title("Position Lambda Convergence Rate - Cutoff, " + str(positionPercentile) + "%")


if plotVelocityConvergenceRates or plotPositionConvergenceRates:
    plt.show()
