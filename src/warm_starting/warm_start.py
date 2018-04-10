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

from Box2D import (b2World, b2LoopShape, b2Vec2, b2ContactListener)

from gen_world import GenClusteredCirclesRegion, create_fixed_box
from sim_types import BodyData



# ----- Parameters -----
# Number of bodies in world
N = 100
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
# World lower left point
p_ll = (xlow, ylow)
# World upper right point
p_hr = (xhi, yhi)
# Grid x-resolution
xRes = 0.5
# Grid y-resolution
yRes = 0.5
# Support radius
h = 3

# Create world in case model needs it
world = b2World(gravity=(0, -10), doSleep=False)

# Choose a model
#model = NoWarmStartModel()
#model = BuiltinWarmStartModel()
#model = BadModel()
#model = RandomModel(0)
#model = ParallelWorldModel(world)
#model = CopyWorldModel()
model = IdentityGridModel(world, p_ll, p_hr, xRes, yRes, h)

# Iteration counter plots
plotIterationCounters = False

# Convergence rate plots
plotConvergenceRates = True
# Choose convergence rate data - 1 for velocity lambda two norms, 2 for velocity lambda infinity norms, 3 for position lambdas
convergenceRateData = 1
# Limit on percentage of contributors left for cutoff (see plot)
limit = 0.2



# ----- Warm-Starting Listener -----
class WarmStartListener(b2ContactListener):
    def __init__(self, model):
        super(WarmStartListener, self).__init__()

        self.model = model

    def PreSolve(self, contact, old_manifold):
        predictions = self.model.Predict(contact)

        # Match predictions to manifold points
        m = contact.manifold
        for point in m.points:
            for pred in predictions:
                id, normal, tangent = pred
                if id.key == point.id.key:
                    point.normalImpulse = normal
                    point.tangentImpulse = tangent



# ----- World Creation -----
world.enableContinuous   = False
world.subStepping        = False
world.enableWarmStarting = True
world.convergenceRates   = plotConvergenceRates

world.velocityThreshold = velocityThreshold
world.positionThreshold = positionThreshold

# Create and attach listener
world.contactListener = WarmStartListener(model)

# Create static box
ground = create_fixed_box(world, p_ll=b2Vec2(xlow, ylow), p_hr=b2Vec2(xhi, yhi))

# Populate the world
gen = GenClusteredCirclesRegion(world, seed=seed)

gen.fill(N, b2Vec2(xlow,ylow),  b2Vec2(xhi,yhi), (1, 1), sigma_coef)

b_ix = 0
for b in world.bodies:
    b.userData.id = b_ix
    b_ix += 1



# ----- Run -----
totalVelocityIterations = []
totalPositionIterations = []
contactsSolved = []
times = []
convergenceRates = []
for i in range(steps):
    print("step", i)

    # Tell the model to take a step
    step = time.time()
    model.Step(world, timeStep, velocityIterations, positionIterations)

    # Tell the world to take a step
    world.Step(timeStep, velocityIterations, positionIterations)
    world.ClearForces()

    step = time.time() - step
    times.append(step)

    # Extract and store profiling data
    profile = world.GetProfile()

    totalVelocityIterations.append(profile.velocityIterations)
    totalPositionIterations.append(profile.positionIterations)
    contactsSolved.append(profile.contactsSolved)

    if plotConvergenceRates:
        if convergenceRateData == 1:
            convergenceRates.append(profile.velocityLambdaTwoNorms)
        if convergenceRateData == 2:
            convergenceRates.append(profile.velocityLambdaInfNorms)
        if convergenceRateData == 3:
            convergenceRates.append(profile.positionLambdas)

    print("Contacts: %d, vel_iter: %d, pos_iter: %d" % (profile.contactsSolved, profile.velocityIterations, profile.positionIterations))



# ----- Process profiling data -----
velocityTotal  = np.sum(totalVelocityIterations)
velocityMean   = np.mean(totalVelocityIterations)
velocityMedian = np.median(totalVelocityIterations)
velocityStd    = np.std(totalVelocityIterations)

positionTotal  = np.sum(totalPositionIterations)
positionMean   = np.mean(totalPositionIterations)
positionMedian = np.median(totalPositionIterations)
positionStd    = np.std(totalPositionIterations)

print("\nVelocity: \nTotal   = %d \nAverage = %.2f \nMedian  = %d \nStd     = %.2f" % (velocityTotal, velocityMean, velocityMedian, velocityStd))
print("\nPosition: \nTotal   = %d \nAverage = %.2f \nMedian  = %d \nStd     = %.2f" % (positionTotal, positionMean, positionMedian, positionStd))

if plotIterationCounters:
    velocityIterationsPerContact = [0 if c == 0 else v / c for c, v in zip(contactsSolved, totalVelocityIterations)]
    positionIterationsPerContact = [0 if c == 0 else p / c for c, p in zip(contactsSolved, totalPositionIterations)]

    # Determine when collissions start happening
    start = 0
    while contactsSolved[start] == 0:
        start += 1

if plotConvergenceRates:
    # Pad convergence rates to be same length using NaNs
    pad = len(max(convergenceRates, key=len))
    convergenceRatesArray = np.array([i + [np.NaN]*(pad-len(i)) for i in convergenceRates])

    tenPercent      = np.nanpercentile(convergenceRatesArray, 10, axis=0)
    firstQuantiles  = np.nanpercentile(convergenceRatesArray, 25, axis=0)
    secondQuantiles = np.nanpercentile(convergenceRatesArray, 50, axis=0)
    thirdQuantiles  = np.nanpercentile(convergenceRatesArray, 75, axis=0)
    ninetyPercent   = np.nanpercentile(convergenceRatesArray, 90, axis=0)

    # Count the number of contributors for each iteration
    counts = np.abs(np.sum(np.isnan(convergenceRatesArray), axis=0) - steps)

    # Determine when to cutoff convergence rate plots
    end = 0
    while counts[end] > steps * limit:
        end += 1



# ----- Plot stuff -----
def pretty(s):
    return '{0:.0E}'.format(s)

titleStats = "N = " + str(N) + ", dt = " + pretty(timeStep) + "\n"
titleStats += "vel_iter = " + pretty(velocityIterations) + ", vel_thres = " + pretty(world.velocityThreshold) + "\n"
titleStats += "pos_iter = " + pretty(positionIterations) + ", pos_thres = " + pretty(world.positionThreshold)


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
    ax1.semilogy(firstQuantiles)
    ax1.semilogy(secondQuantiles)
    ax1.semilogy(thirdQuantiles)
    ax1.legend(["25", "50", "75"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(title)
    ax1.set_title("Convergence Rate - All iterations")

    # Counters
    ax1 = fig.add_subplot(222)
    ax1.plot(counts)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

    # Limited convergence rates
    ax1 = fig.add_subplot(223)
    ax1.semilogy(firstQuantiles)
    ax1.semilogy(secondQuantiles)
    ax1.semilogy(thirdQuantiles)
    ax1.set_xlim([0, end])
    ax1.legend(["25", "50", "75"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(title)
    ax1.set_title("Convergence Rate - " + str(int(limit*100)) + "% cutoff")

    # Limited convergence rates
    ax1 = fig.add_subplot(224)
    ax1.semilogy(tenPercent)
    ax1.semilogy(secondQuantiles)
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
    ln1 = ax1.plot(range(steps), totalVelocityIterations, 'c', label="total_iter")
    ax1.set_xlim([start-5, steps])
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of iterations")
    ax1.tick_params('y', colors='c')

    ax2 = ax1.twinx()
    ln2 = ax2.plot(range(steps), velocityIterationsPerContact, 'm', label="contact_iter")
    ax2.set_ylabel('Number of iterations per contact')
    ax2.tick_params('y', colors='m')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    ax1.set_title("Velocity iterations numbers")


    # Position iterations
    ax1 = fig.add_subplot(224)
    ln1 = ax1.plot(range(steps), totalPositionIterations, 'c', label="total_iter")
    ax1.set_xlim([start-5, steps])
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of iterations")
    ax1.tick_params('y', colors='c')

    ax2 = ax1.twinx()
    ln2 = ax2.plot(range(steps), positionIterationsPerContact, 'm', label="contact_iter")
    ax2.set_ylabel('Number of iterations per contact')
    ax2.tick_params('y', colors='m')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    ax1.set_title("Position iterations numbers")


    # Times
    ax1 = fig.add_subplot(221)
    ax1.plot(range(steps), times)
    ax1.set_xlim([start-5, steps])
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Step time")
    ax1.set_title("Time taken for each step")


    # Contacts
    ax1 = fig.add_subplot(222)
    ax1.plot(range(steps), contactsSolved)
    ax1.set_xlim([start-5, steps])
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of contacts")
    ax1.set_title("Contact numbers for each step")


if plotConvergenceRates or plotIterationCounters:
    plt.show()
