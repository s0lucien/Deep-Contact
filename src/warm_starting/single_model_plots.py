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

from ..gen_world import new_confined_clustered_circles_world
from .warm_start import run_world


# ----- Parameters -----
# Number of bodies in world
N = 100
# Seed to use for body generator
seed = 1337
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
xRes = 0.8
# Grid y-resolution
yRes = 0.8
# Support radius
h = 1

# Create world in case model needs it
world = b2World()
# Fill world with static box and circles
new_confined_clustered_circles_world(world, N, b2Vec2(p_ll), b2Vec2(p_ur), (1, 1), sigma_coef, seed)

# Choose a model
#model = NoWarmStartModel()
#model = BuiltinWarmStartModel()
#model = BadModel()
#model = RandomModel(0)
#model = ParallelWorldModel(world)
#model = CopyWorldModel()
model = IdentityGridModel(world, p_ll, p_ur, xRes, yRes, h)

# Iteration counter plots
plotIterationCounters = True

# Convergence rate plots
plotConvergenceRates = True
# Choose convergence rate data
# 1 for velocity lambda two norms
# 2 for velocity lambda infinity norms
# 3 for position lambdas
convergenceRateData = 1
# Limit on percentage of contributors left for cutoff (see plot)
limit = 0.2



# ----- Run simulation -----
result = run_world(world, model, timeStep, steps,
                   velocityIterations, positionIterations, velocityThreshold, positionThreshold,
                   iterations=True, convergenceRates=True, quiet=False)

totalVelocityIterations = result["totalVelocityIterations"]
totalPositionIterations = result["totalPositionIterations"]
totalStepTimes = result["totalStepTimes"]
contactsSolved = result["contactsSolved"]
iteratorCounts = result["iteratorCounts"]



# ----- Process data -----
velocityTotal  = np.sum(totalVelocityIterations)
velocityMean   = np.mean(totalVelocityIterations)
velocityMedian = np.median(totalVelocityIterations)
velocityStd    = np.std(totalVelocityIterations)

positionTotal  = np.sum(totalPositionIterations)
positionMean   = np.mean(totalPositionIterations)
positionMedian = np.median(totalPositionIterations)
positionStd    = np.std(totalPositionIterations)

print("\nVelocity:")
print("Total   = %d"   % velocityTotal)
print("Average = %.2f" % velocityMean)
print("Median  = %d"   % velocityMedian)
print("Std     = %.2f" % velocityStd)

print("\nPosition:")
print("Total   = %d"   % positionTotal)
print("Average = %.2f" % positionMean)
print("Median  = %d"   % positionMedian)
print("Std     = %.2f" % positionStd)

if plotIterationCounters:
    # Determine when collissions start happening
    start = 0
    while contactsSolved[start] == 0:
        start += 1

    velocityIterationsPerContact = [0 if c == 0 else v / c for c, v in zip(result["contactsSolved"], result["totalVelocityIterations"])]
    positionIterationsPerContact = [0 if c == 0 else p / c for c, p in zip(result["contactsSolved"], result["totalPositionIterations"])]

if plotConvergenceRates:
    # Determine when to cutoff convergence rate plots
    end = 0
    while iteratorCounts[end] > steps * limit:
        end += 1

    if convergenceRateData == 1:
        convergenceData = result["velocityLambdaTwoNorms"]
    if convergenceRateData == 2:
        convergenceData = result["velocityLambdaInfNorms"]
    if convergenceRateData == 3:
        convergenceData = result["positionLambdas"]

    # Pad convergence rates to be same length using NaNs
    pad = len(max(convergenceData, key=len))
    convergenceRatesArray = np.array([i + [np.NaN]*(pad-len(i)) for i in convergenceData])

    tenPercent         = np.nanpercentile(convergenceRatesArray, 10, axis=0)
    twentyfivePercent  = np.nanpercentile(convergenceRatesArray, 25, axis=0)
    fiftyPercent       = np.nanpercentile(convergenceRatesArray, 50, axis=0)
    seventyfivePercent = np.nanpercentile(convergenceRatesArray, 75, axis=0)
    ninetyPercent      = np.nanpercentile(convergenceRatesArray, 90, axis=0)



# ----- Plot stuff -----
# Make an overall plot title with a few stats
def pretty(s):
    return '{0:.0E}'.format(s)

titleStats = "N = " + str(N) + ", dt = " + pretty(timeStep) + "\n"
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
    ax1.semilogy(twentyfivePercent)
    ax1.semilogy(fiftyPercent)
    ax1.semilogy(seventyfivePercent)
    ax1.legend(["25", "50", "75"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(title)
    ax1.set_title("Convergence Rate - All iterations")

    # Counters
    ax1 = fig.add_subplot(222)
    ax1.plot(iteratorCounts)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

    # Limited convergence rates
    ax1 = fig.add_subplot(223)
    ax1.semilogy(twentyfivePercent)
    ax1.semilogy(fiftyPercent)
    ax1.semilogy(seventyfivePercent)
    ax1.set_xlim([0, end])
    ax1.legend(["25", "50", "75"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(title)
    ax1.set_title("Convergence Rate - " + str(int(limit*100)) + "% cutoff")

    # Limited convergence rates - different quantiles
    ax1 = fig.add_subplot(224)
    ax1.semilogy(tenPercent)
    ax1.semilogy(fiftyPercent)
    ax1.semilogy(ninetyPercent)
    ax1.set_xlim([0, end])
    ax1.legend(["10", "50", "90"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(title)
    ax1.set_title("Convergence Rate - " + str(int(limit*100)) + "% cutoff, different quantiles")


# Iteration plots
if plotIterationCounters:
    fig = plt.figure()
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


if plotConvergenceRates or plotIterationCounters:
    plt.show()
