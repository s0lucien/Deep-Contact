import time
import numpy as np
import matplotlib.pyplot as plt

from .builtin_warmstart_model import BuiltinWarmStartModel
from .bad_model import BadModel
from .random_model import RandomModel
from .parallel_world_model import ParallelWorldModel
from .copy_world_model import CopyWorldModel
from .identity_grid_model import IdentityGridModel
from .nn_model import NNModel

from Box2D import (b2World, b2Vec2)

from ..gen_world import new_confined_clustered_circles_world
from .warm_start import run_world
from ..tensorflow.cnn import CNN


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


# Create world in case model needs it
world = b2World()
# Fill world with static box and circles
new_confined_clustered_circles_world(world, nBodies, b2Vec2(xlow, ylow), b2Vec2(xhi, yhi), r, sigma_coef, seed)

# Choose a model
#model = None
model = BuiltinWarmStartModel()
#model = BadModel()
#model = RandomModel(0)
#model = ParallelWorldModel(world)
#model = CopyWorldModel()
#model = IdentityGridModel((xlow, ylow), (xhi, yhi), 0.25, 0.25, 1)
#model = NNModel(CNN({}))


# Iteration counter plots
plotIterationCounters = True
# Velocity convergence rate plots
plotVelocityConvergenceRates = True
# Position convergence rate plots
plotPositionConvergenceRates = False
# Limit on percentage of contributors left for cutoff (see convergence plots)
limit = 0.2

# Print various iteration numbers as simulation is running
printing = True
# Show visualization of world as simulation is running
# note: significantly slower
visualize = False


# ----- Run simulation -----
conv = plotVelocityConvergenceRates | plotPositionConvergenceRates
iter = plotIterationCounters
result = run_world(world, timeStep, steps,
                   velocityIterations, positionIterations,
                   velocityThreshold=velocityThreshold, positionThreshold=positionThreshold,
                   model=model, iterations=iter, convergenceRates=conv,
                   quiet=not printing, visualize=visualize)



# ----- Process data -----
totalStepTimes = result["totalStepTimes"]
contactsSolved = result["contactsSolved"]


if plotIterationCounters:
    totalVelocityIterations = result["totalVelocityIterations"]
    totalPositionIterations = result["totalPositionIterations"]

    # Determine when collissions start happening
    start = 0
    while contactsSolved[start] == 0:
        start += 1

    velocityIterationsPerContact = [0 if c == 0 else v / c for c, v in zip(result["contactsSolved"], result["totalVelocityIterations"])]
    positionIterationsPerContact = [0 if c == 0 else p / c for c, p in zip(result["contactsSolved"], result["totalPositionIterations"])]


if plotVelocityConvergenceRates:
    velocityLambdaTwoNorms = result["velocityLambdaTwoNorms"]
    velocityLambdaInfNorms = result["velocityLambdaInfNorms"]
    velocityIteratorCounts = result["velocityIteratorCounts"]

    # Determine when to cutoff convergence rate plots
    velocityEnd = 0
    while velocityEnd < len(velocityIteratorCounts) and velocityIteratorCounts[velocityEnd] > steps * limit:
        velocityEnd += 1

    # Pad convergence rates to be same length using NaNs
    pad = len(max(velocityLambdaTwoNorms, key=len))
    velocityTwoArray = np.array([i + [np.NaN]*(pad-len(i)) for i in velocityLambdaTwoNorms])
    velocityInfArray = np.array([i + [np.NaN]*(pad-len(i)) for i in velocityLambdaInfNorms])

    # Determine quantiles for lambda two-norms
    velocityTwoMin = np.nanmin(velocityTwoArray, axis=0)
    velocityTwoFst = np.nanpercentile(velocityTwoArray, 25, axis=0)
    velocityTwoSnd = np.nanpercentile(velocityTwoArray, 50, axis=0)
    velocityTwoThr = np.nanpercentile(velocityTwoArray, 75, axis=0)
    velocityTwoMax = np.nanmax(velocityTwoArray, axis=0)

    # Determine quantiles for lambda inf-norms
    velocityInfMin = np.nanmin(velocityInfArray, axis=0)
    velocityInfFst = np.nanpercentile(velocityInfArray, 25, axis=0)
    velocityInfSnd = np.nanpercentile(velocityInfArray, 50, axis=0)
    velocityInfThr = np.nanpercentile(velocityInfArray, 75, axis=0)
    velocityInfMax = np.nanmax(velocityInfArray, axis=0)


if plotPositionConvergenceRates:
    positionLambdas = result["positionLambdas"]
    positionIteratorCounts = result["positionIteratorCounts"]

    # Determine when to cutoff convergence rate plots
    positionEnd = 0
    while positionEnd < len(positionIteratorCounts) and positionIteratorCounts[positionEnd] > steps * limit:
        positionEnd += 1

    # Pad convergence rates to be same length using NaNs
    pad = len(max(positionLambdas, key=len))
    positionArray = np.array([i + [np.NaN]*(pad-len(i)) for i in positionLambdas])

    # Determine quantiles for lambdas
    positionMin = np.nanmin(positionArray, axis=0)
    positionFst = np.nanpercentile(positionArray, 25, axis=0)
    positionSnd = np.nanpercentile(positionArray, 50, axis=0)
    positionThr = np.nanpercentile(positionArray, 75, axis=0)
    positionMax = np.nanmax(positionArray, axis=0)



# ----- Plot stuff -----
# Make an overall plot title with a few stats
def pretty(s):
    return '{0:.0E}'.format(s)

titleStats = "nBodies = " + str(nBodies) + ", dt = " + pretty(timeStep) + ", steps = " + pretty(steps) + "\n"
titleStats += "vel_iter = " + pretty(velocityIterations)
titleStats += ", vel_thres = " + pretty(world.velocityThreshold) + "\n"
titleStats += "pos_iter = " + pretty(positionIterations)
titleStats += ", pos_thres = " + pretty(world.positionThreshold)

# Iteration plots
if plotIterationCounters:
    fig = plt.figure("Iterations")
    fig.suptitle("Iteration Counters" + "\n" + titleStats)

    # Velocity iterations
    ax1 = fig.add_subplot(223)
    ln1 = ax1.plot(totalVelocityIterations, 'c', label="total_iter")
    ax1.set_xlim([start-5, steps])
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of iterations")
    ax1.tick_params('y', colors='c')

    ax2 = ax1.twinx()
    ln2 = ax2.plot(velocityIterationsPerContact, 'm', label="contact_iter")
    ax2.set_ylabel('Number of iterations per contact')
    ax2.tick_params('y', colors='m')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    ax1.set_title("Velocity iterations numbers")

    # Position iterations
    ax1 = fig.add_subplot(224)
    ln1 = ax1.plot(totalPositionIterations, 'c', label="total_iter")
    ax1.set_xlim([start-5, steps])
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of iterations")
    ax1.tick_params('y', colors='c')

    ax2 = ax1.twinx()
    ln2 = ax2.plot(positionIterationsPerContact, 'm', label="contact_iter")
    ax2.set_ylabel('Number of iterations per contact')
    ax2.tick_params('y', colors='m')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    ax1.set_title("Position iterations numbers")

    # Times
    ax1 = fig.add_subplot(221)
    ax1.plot(totalStepTimes)
    ax1.set_xlim([start-5, steps])
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Step time")
    ax1.set_title("Time taken for each step")

    # Contacts
    ax1 = fig.add_subplot(222)
    ax1.plot(contactsSolved)
    ax1.set_xlim([start-5, steps])
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of contacts")
    ax1.set_title("Contact numbers for each step")


# Velocity convergence rate plots
if plotVelocityConvergenceRates:
    fig = plt.figure("Velocity Convergence")
    fig.suptitle("Velocity Convergence Rates\n" + titleStats)

    # Full convergence rates
    ax1 = fig.add_subplot(221)
    ax1.semilogy(velocityTwoMax, "r")
    ax1.semilogy(velocityTwoThr, "g")
    ax1.semilogy(velocityTwoSnd, "b")
    ax1.semilogy(velocityTwoFst, "g")
    ax1.semilogy(velocityTwoMin, "r")
    ax1.legend(["Max", "75", "50", "25", "Min"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Velocity Lambda Two-Norms")
    ax1.set_title("Lambda Two-Norm Convergence Rate - All iterations")

    # Counters
    ax1 = fig.add_subplot(222)
    ax1.plot(velocityIteratorCounts)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

    # Limited convergence rates - Two-norm
    ax1 = fig.add_subplot(223)
    ax1.semilogy(velocityTwoMax, "r")
    ax1.semilogy(velocityTwoThr, "g")
    ax1.semilogy(velocityTwoSnd, "b")
    ax1.semilogy(velocityTwoFst, "g")
    ax1.semilogy(velocityTwoMin, "r")
    ax1.set_xlim([0, velocityEnd])
    ax1.legend(["Max", "75", "50", "25", "Min"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Velocity Lambda Two-Norms")
    ax1.set_title("Lambda Two-norm Convergence Rate - " + str(int(limit*100)) + "% cutoff")

    # Limited convergence rates - Inf-norm
    ax1 = fig.add_subplot(224)
    ax1.semilogy(velocityInfMax, "r")
    ax1.semilogy(velocityInfThr, "g")
    ax1.semilogy(velocityInfSnd, "b")
    ax1.semilogy(velocityInfFst, "g")
    ax1.semilogy(velocityInfMin, "r")
    ax1.set_xlim([0, velocityEnd])
    ax1.legend(["Max", "75", "50", "25", "Min"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Velocity Lambda Inf-Norms")
    ax1.set_title("Lambda Inf-Norm Convergence Rate - " + str(int(limit*100)) + "% cutoff")


# Position convergence rate plots
if plotPositionConvergenceRates:
    fig = plt.figure("Position Convergence")
    fig.suptitle("Position Convergence Rates\n" + titleStats)

    # Full convergence rates
    ax1 = fig.add_subplot(221)
    ax1.semilogy(positionMax, "r")
    ax1.semilogy(positionThr, "g")
    ax1.semilogy(positionSnd, "b")
    ax1.semilogy(positionFst, "g")
    ax1.semilogy(positionMin, "r")
    ax1.legend(["Max", "75", "50", "25", "Min"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Position Lambdas")
    ax1.set_title("Lambda Convergence Rate - All iterations")

    # Counters
    ax1 = fig.add_subplot(222)
    ax1.plot(positionIteratorCounts)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

    # Limited convergence rates
    ax1 = fig.add_subplot(223)
    ax1.semilogy(positionMax, "r")
    ax1.semilogy(positionThr, "g")
    ax1.semilogy(positionSnd, "b")
    ax1.semilogy(positionFst, "g")
    ax1.semilogy(positionMin, "r")
    ax1.set_xlim([0, positionEnd])
    ax1.legend(["Max", "75", "50", "25", "Min"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Position Lambdas")
    ax1.set_title("Lambda Convergence Rate - " + str(int(limit*100)) + "% cutoff")


if plotIterationCounters or plotVelocityConvergenceRates or plotPositionConvergenceRates:
    plt.show()
