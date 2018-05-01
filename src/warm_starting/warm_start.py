import time
import numpy as np

from Box2D import (b2ContactListener)
import cv2
import matplotlib.pyplot as plt

from ..opencv_draw import OpencvDrawFuncs
from ..xml_writing.b2d_2_xml import XMLExporter

# Warm-Starting Listener
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

#
def run_world(world, timeStep, steps,
              velocityIterations, positionIterations,
              velocityThreshold=0, positionThreshold=1000,
              model=None, iterations=False, convergenceRates=False,
              storeAsXML=False, path="",
              quiet=True, visualize=False):

    # ----- Setup -----
    # Enable/disable convergence rates
    world.convergenceRates = convergenceRates

    # Set iteration thresholds
    world.velocityThreshold = velocityThreshold
    world.positionThreshold = positionThreshold

    # Create and attach listener if given a model
    if model:
        world.contactListener = WarmStartListener(model)
    else:
        world.warmStarting = False

    # Define a drawer if set
    if visualize:
        drawer = OpencvDrawFuncs(w=640, h=640, ppm=10)
        drawer.install()

    # Initiate the XML exporter if set
    if storeAsXML:
        exp = XMLExporter(world, path)

    # We store the performance data in a dictionary
    result = {}

    # ----- Run World -----
    totalStepTimes          = []
    contactsSolved          = []
    totalVelocityIterations = []
    totalPositionIterations = []
    velocityLambdaTwoNorms  = []
    velocityLambdaInfNorms  = []
    positionLambdas         = []
    for i in range(steps):
        if not quiet:
            print("step", i)

        # Start step timer
        step = time.time()

        # Tell the model to take a step
        if model:
            model.Step(world, timeStep, velocityIterations, positionIterations)

        # Draw the world
        if visualize:
            drawer.clear_screen()
            drawer.draw_world(world)

            cv2.imshow('World', drawer.screen)
            cv2.waitKey(25)

        # Tell the world to take a step
        world.Step(timeStep, velocityIterations, positionIterations)
        world.ClearForces()

        # Determine total step time
        step = time.time() - step
        totalStepTimes.append(step)

        # Store world as XML if set
        if storeAsXML:
            world.userData.tick()
            exp.save_snapshot()

        # Extract and store profiling data
        profile = world.GetProfile()

        contactsSolved.append(profile.contactsSolved)

        if iterations:
            totalVelocityIterations.append(profile.velocityIterations)
            totalPositionIterations.append(profile.positionIterations)

        if convergenceRates:
            velocityLambdaTwoNorms.append(profile.velocityLambdaTwoNorms)
            velocityLambdaInfNorms.append(profile.velocityLambdaInfNorms)
            positionLambdas.append(profile.positionLambdas)

        if not quiet:
            print("Contacts: %d, vel_iter: %d, pos_iter: %d" %
                  (profile.contactsSolved, profile.velocityIterations, profile.positionIterations))


    # Print results
    if not quiet:
        if iterations:
            print("\nVelocity:")
            print("Total   = %d"   % np.sum(totalVelocityIterations))
            print("Average = %.2f" % np.mean(totalVelocityIterations))
            print("Median  = %d"   % np.median(totalVelocityIterations))
            print("Std     = %.2f" % np.std(totalVelocityIterations))

            print("\nPosition:")
            print("Total   = %d"   % np.sum(totalPositionIterations))
            print("Average = %.2f" % np.mean(totalPositionIterations))
            print("Median  = %d"   % np.median(totalPositionIterations))
            print("Std     = %.2f" % np.std(totalPositionIterations))


    # Store results
    result["totalStepTimes"] = totalStepTimes
    result["contactsSolved"] = contactsSolved

    if iterations:
        result["totalVelocityIterations"] = totalVelocityIterations
        result["totalPositionIterations"] = totalPositionIterations

    if convergenceRates:
        result["velocityLambdaTwoNorms"] = velocityLambdaTwoNorms
        result["velocityLambdaInfNorms"] = velocityLambdaInfNorms
        result["positionLambdas"] = positionLambdas

        # Count the number of contributors for each velocity iteration
        iterations = [len(l) for l in velocityLambdaTwoNorms]
        result["velocityIteratorCounts"] = [np.sum([l >= i for l in iterations]) for i in range(max(iterations))]

        # Count the number of contributors for each position iteration
        iterations = [len(l) for l in positionLambdas]
        result["positionIteratorCounts"] = [np.sum([l >= i for l in iterations]) for i in range(max(iterations))]

    # Return results
    return result
