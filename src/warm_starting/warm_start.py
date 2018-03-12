import time
import numpy as np

from .no_warmstart_model import NoWarmStartModel
from .builtin_warmstart_model import BuiltinWarmStartModel
from .bad_model import BadModel
from .random_model import RandomModel
from .parallel_world_model import ParallelWorldModel
from .copy_world_model import CopyWorldModel

from Box2D import (b2World)
from Box2D import (b2LoopShape)
from Box2D import (b2ContactListener)
from Box2D import (b2Vec2)

from ..gen_world import GenClusteredCirclesWorld
from ..sim_types import BodyData

# ----- World Creation -----
# Create world
world = b2World(gravity=(0, -10), doSleep=False)

world.enableContinuous   = False
world.subStepping        = False
world.enableWarmStarting = True

# Create static box
xlow, xhi = -30, 30
ylow, yhi = 0, 60

ground = world.CreateBody(
    shapes=b2LoopShape(
        vertices=[(xhi, ylow), (xhi, yhi), (xlow, yhi), (xlow, ylow)]
    )
)

# Populate the world
N = 100
seed = 100
gen = GenClusteredCirclesWorld(world, seed=seed)
sigma_coef = 1.2
gen.new(N, b2Vec2(xlow,ylow),  b2Vec2(xhi,yhi), 1, 1, sigma_coef)

b_ix = 0
for b in world.bodies:
    b.userData = BodyData(b_ix)
    b_ix += 1



# ----- Warm-Starting Listener -----
class WarmStartListener(b2ContactListener):
    def __init__(self, model):
        super(WarmStartListener, self).__init__()

        self.model = model

    def PreSolve(self, contact, old_manifold):
        #print("Pre")
        predictions = self.model.Predict(contact)

        # Match predictions to manifold points
        m = contact.manifold
        for point in m.points:
            for pred in predictions:
                id, normal, tangent = pred
                if id.key == point.id.key:
                    point.normalImpulse = normal
                    point.tangentImpulse = tangent



# ----- Run -----
timeStep = 1.0 / 100
velocityIterations = 20000
positionIterations = 10000
world.velocityThreshold = 10**-5
world.positionThreshold = 10**-6
steps = 1000

# Choose a model
#model = NoWarmStartModel()
#model = BuiltinWarmStartModel()
#model = BadModel()
#model = RandomModel(0)
#model = ParallelWorldModel(world)
model = CopyWorldModel()

# Create and attach listener
world.contactListener = WarmStartListener(model)


# Run the simulation
totalVelocityIterations = []
totalPositionIterations = []
contactsSolved = []
totalContacts = []
times = []
for i in range(steps):
    print("step", i)

    # Tell the model to take a step
    step = time.time()
    model.Step(world, timeStep, velocityIterations, positionIterations)

    # Tell the world to take a step
    world.Step(timeStep, velocityIterations, positionIterations)
    world.ClearForces()
    step = time.time() - step

    # Do stuff with profiling data
    profile = world.GetProfile()

    totalVelocityIterations.append(profile.velocityIterations)
    totalPositionIterations.append(profile.positionIterations)
    contactsSolved.append(profile.contactsSolved)

    times.append(step)

    nc = 0
    for c in world.contacts:
        if c.touching:
            nc += 1
    totalContacts.append(nc)

    print("Contacts: %d, solved: %d, vel_iter: %d, pos_iter: %d" % (nc, profile.contactsSolved, profile.velocityIterations, profile.positionIterations))


# Process stuff
velocityIterationsPerContact = [0 if c == 0 else v / c for c, v in zip(contactsSolved, totalVelocityIterations)]
positionIterationsPerContact = [0 if c == 0 else p / c for c, p in zip(contactsSolved, totalPositionIterations)]

velocityTotal  = np.sum(totalVelocityIterations)
velocityMean   = np.mean(totalVelocityIterations)
velocityMedian = np.median(totalVelocityIterations)
velocityStd    = np.std(totalVelocityIterations)

positionTotal  = np.sum(totalPositionIterations)
positionMean   = np.mean(totalPositionIterations)
positionMedian = np.median(totalPositionIterations)
positionStd    = np.std(totalPositionIterations)

print("\nVelocity: \nTotal   = %d \nAverage = %.2f \nMedian  = %d \nStd     = %.2f" % (velocityTotal, velocityMean, velocityMedian, velocityStd))
print("\nPosition: \nTotal   = %d \nAverage = %.2f \nMedian  = %d \nStd     = %.2f" % (positionTotal, positionMean, positionMedian, velocityStd))


# Plot stuff
import matplotlib.pyplot as plt

start = 0

def pretty(s):
    return '{0:.0E}'.format(s)

fig = plt.figure()
title = "N = " + str(N) + ", dt = " + pretty(timeStep) + ", vel_iter = " + pretty(velocityIterations) + ", pos_iter = " + pretty(positionIterations) + ", vel_thres = " + pretty(world.velocityThreshold) + ", pos_thres = " + pretty(world.positionThreshold)
fig.suptitle(title)

# Velocity iterations
ax1 = fig.add_subplot(223)
ln1 = ax1.plot(range(steps), totalVelocityIterations, 'c', label="total_iter")
ax1.set_xlim([start, steps])
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
ax1.set_xlim([start, steps])
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
ax1.set_xlim([start, steps])
ax1.set_xlabel("Step")
ax1.set_ylabel("Step time")
ax1.set_title("Time taken for each step")


# Contacts
ax1 = fig.add_subplot(222)
ln1 = ax1.plot(range(steps), totalContacts, 'c', label="total")
ax1.set_xlim([start, steps])
ax1.set_xlabel("Step")
ax1.set_ylabel("Total number of contacts")
ax1.tick_params('y', colors='c')

ax2 = ax1.twinx()
ln2 = ax2.plot(range(steps), contactsSolved, 'm', label="solved")
ax2.set_ylabel('Number of solved contacts')
ax2.tick_params('y', colors='m')

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="lower center")
ax1.set_title("Contact numbers for each step")


plt.show()
