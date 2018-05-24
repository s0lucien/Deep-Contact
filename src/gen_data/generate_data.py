import cv2
import os

from Box2D import (b2World, b2Vec2, b2ContactListener)

from ..gen_world import new_confined_clustered_circles_world
from ..sim_types import SimData
from ..xml_writing.b2d_2_xml import XMLExporter
from ..opencv_draw import OpencvDrawFuncs


# ----- Parameters -----
# Number of bodies in world
nBodies = 50
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
steps = 800

# Decides whether to store configurations without any contacts
skipContactless = False
# Print various iteration numbers as simulation is running
quiet = True


# ----- Misc -----
# B2ContactListener for recording contacts and impulses
class ContactListener(b2ContactListener):
    def __init__(self, exporter: XMLExporter):
        super(ContactListener, self).__init__()

        self.xml_exp = exporter
        self.reset()

    # Reset the counter in preparation for a new step
    def reset(self):
        self.counter = 0

    # Store all pre-solve contact information
    def PreSolve(self, contact, _):
        # We give the contact an index so that we can recognize it later
        contact.userData = self.counter
        self.counter += contact.manifold.pointCount

        self.xml_exp.snapshot_contact(contact)

    # Store post-solve impulses
    def PostSolve(self, contact, impulse):
        self.xml_exp.snapshot_impulse(contact, impulse)


# ----- Data generation -----
if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-s', '--seeds', type='int', default=100,
                      dest='seeds')
    parser.add_option('-p', '--path', dest='path')
    parser.add_option('-V', '--visualize', action='store_true',
                      dest='visualize')

    options, _ = parser.parse_args()

    seeds = range(options.seeds)
    path = options.path
    visualize = options.visualize

    for i in range(len(seeds)):
        # image generation
        drawer = OpencvDrawFuncs(w=300, h=300, ppm=10)
        drawer.install()

        seed = seeds[i]
        print("Running world %d of %d" % (i+1, len(seeds)))

        # Create world
        world = b2World()
        world.userData = SimData(
            name=str(seed),
            d_t=timeStep,
            vel_iter=velocityIterations,
            pos_iter=positionIterations,
            vel_thres=velocityThreshold,
            pos_thres=positionThreshold
        )

        # Fill world with static box and circles
        new_confined_clustered_circles_world(
            world, nBodies, b2Vec2(xlow, ylow), b2Vec2(xhi, yhi), r, sigma_coef, seed)

        # Set iteration thresholds
        world.velocityThreshold = velocityThreshold
        world.positionThreshold = positionThreshold

        # Initialize XML exporter
        xml_exp = XMLExporter(world, path)

        # Initialize contact listener
        listener = ContactListener(xml_exp)
        world.contactListener = listener

        # Run simulation
        for step in range(steps):
            if not quiet:
                print("step", step)

            # Reset the contact listener
            listener.reset()

            # Reset xml exporter and take snapshot of bodies
            xml_exp.reset()
            xml_exp.snapshot_bodies()

            # Tell the world to take a step
            world.Step(timeStep, velocityIterations, positionIterations)
            world.userData.tick()
            world.ClearForces()

            # Draw the world
            if visualize:
                drawer.clear_screen()
                drawer.draw_world(world)

                cv2.imwrite(
                    os.path.join(path,
                                 '{}'.format(i),
                                 '{}_{}.png'.format(i, step)),
                    drawer.screen,
                )

                cv2.imshow('World', drawer.screen)
                cv2.waitKey(25)

            # Save the snapshot if wanted
            if not skipContactless or world.GetProfile().contactsSolved > 0:
                xml_exp.save_snapshot()
