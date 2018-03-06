import time
import numpy as np

from vanilla_model import VanillaModel
from bad_model import BadModel
from random_model import RandomModel
from parallel_world_model import ParallelWorldModel
from copy_world_model import CopyWorldModel

from Box2D import (b2World)
from Box2D import (b2_dynamicBody)
from Box2D import (b2FixtureDef)
from Box2D import (b2PolygonShape, b2CircleShape)
from Box2D import (b2ContactListener)



# ----- World Creation -----
# Create world
world = b2World(gravity=(0, -10), doSleep=True)

world.enableContinuous   = False
world.subStepping        = False
world.enableWarmStarting = True

# Create static box
width = 30
world.CreateStaticBody(
    position=(0, 0),
    shapes=[
        b2PolygonShape(box=(0.5, width, (width, 0), 0)),
        b2PolygonShape(box=(0.5, width, (-width, 0), 0)),
        b2PolygonShape(box=(width, 0.5, (0, -width), 0)),
    ],
    userData="0"
)

# Create 'N' objects in world of type 'shape'
circle = b2CircleShape(radius=0.8)
box = b2PolygonShape(box=(0.6, 0.6))
shape = circle

fixture = b2FixtureDef(shape=shape, density=1, restitution=0, friction=0.5)

N = 200 # Number of bodies in simulation
layers = min(2*width, N // (width-2) + (0 if N%width==0 else 1))
count = 1
for y in range(layers):
    for x in range(width-2):
        if count > N:
            break
        world.CreateBody(type=b2_dynamicBody, fixtures=fixture, userData=str(count),
                         position=(width-4-2*x + (1 if y%2==0 else 0), 2*y))
        count += 1



# ----- Warm-Starting -----
class WarmStartListener(b2ContactListener):
    def __init__(self, model):
        super(WarmStartListener, self).__init__()

        self.model = model

    def BeginContact(self, contact):
        #print("Begin")
        pass

    def EndContact(self, contact):
        #print("End")
        pass

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

    def PostSolve(self, contact, impulse):
        #print("Post")
        pass



# ----- Run -----
# Choose a model
#model = VanillaModel()
#model = BadModel()
#model = RandomModel(0)
model = ParallelWorldModel(world)
#model = CopyWorldModel()

# Create and attach listener
world.contactListener = WarmStartListener(model)

# Run the simulation
timeStep = 1.0 / 100
velocityIterations = 20000
positionIterations = 10000
velocityThreshold = 10**-5
positionThreshold = 10**-6

totalVelocityIterations = []
totalPositionIterations = []
contactsSolved = []
totalContacts = []
times = []
steps = 800
for i in range(steps):
    print("step", i)

    # Tell the model to take a step
    step = time.time()
    model.Step(world, timeStep, velocityIterations, positionIterations, velocityThreshold, positionThreshold)

    # Tell the world to take a step
    world.Step(timeStep, velocityIterations, positionIterations, velocityThreshold, positionThreshold)
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

start = 200

def pretty(s):
    return '{0:.0E}'.format(s)

fig = plt.figure()
title = "N = " + str(N) + ", dt = " + pretty(timeStep) + ", vel_iter = " + pretty(velocityIterations) + ", pos_iter = " + pretty(positionIterations) + ", vel_thres = " + pretty(velocityThreshold) + ", pos_thres = " + pretty(positionThreshold)
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
