import time

from model import Model
from bad_model import BadModel
from random_model import RandomModel
from pybox_model_1 import PyboxModel1
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
world.enableWarmStarting = True
world.subStepping        = False

# Create static box
width = 30
world.CreateStaticBody(
    position=(0, 0),
    shapes=[
        b2PolygonShape(box=(0.5, width, (width, 0), 0)),
        b2PolygonShape(box=(0.5, width, (-width, 0), 0)),
        b2PolygonShape(box=(width, 0.5, (0, width), 0)),
        b2PolygonShape(box=(width, 0.5, (0, -width), 0)),
    ],
    userData="0"
)

# Create 'N' objects in world of type 'shape'
circle = b2CircleShape(radius=0.8)
box = b2PolygonShape(box=(0.6, 0.6))
shape = box

fixture = b2FixtureDef(shape=shape, density=1)

N = 604 # Number of bodies in simulation
layers = min(2*width, N // (width-2) + (0 if N%width==0 else 1))
count = 1
for y in range(layers):
    for x in range(width-2):
        if count > N:
            break
        world.CreateBody(type=b2_dynamicBody, fixtures=fixture, userData=str(count),
                         position=(width-4-2*x + (1 if y%2==0 else 0), width-2-2*y))
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
#model = Model()
#model = BadModel()
#model = RandomModel(0)
model = ParallelWorldModel(world)
#model = CopyWorldModel()

# Create and attach listener
world.contactListener = WarmStartListener(model)

# Run the simulation
timeStep = 1.0 / 60
velocityIterations = 10000
positionIterations = 10000
stepTuple = (timeStep, velocityIterations, positionIterations)

totalVelocityIterations = []
totalPositionIterations = []
times = []
steps = 200
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
    nc = 0
    for c in world.contacts:
        if c.touching:
            nc += 1

    # timePerContact = 0 if world.contactCount==0 else profile.collide / world.contactCount
    # results.append(timePerContact)
    # results.append(step)
    if nc == 0:
        totalVelocityIterations.append(0)
        totalPositionIterations.append(0)
    else:
        totalVelocityIterations.append(profile.iterationsVelocity / nc)
        totalPositionIterations.append(profile.iterationsPosition / nc)

    times.append(step)


# Plot stuff
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ln1 = ax1.plot(range(steps), totalVelocityIterations, 'c', label="vel_iter")
#ax1.set_xlim([125, 200])
ax1.set_xlabel("Step")
ax1.set_ylabel("Iterations per contact")
ax1.tick_params('y', colors='c')

ax2 = ax1.twinx()
ln2 = ax2.plot(range(steps), totalPositionIterations, 'm', label="vel_pos")
ax2.set_ylabel('Iterations per contact')
ax2.tick_params('y', colors='m')

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)

fig.tight_layout()
plt.title("Total number of iterations per contact for each step")
plt.show()

plt.figure()
plt.plot(range(steps), times)
plt.xlabel("Step")
plt.ylabel("Step time")
plt.title("Time taken for each step")
plt.show()
