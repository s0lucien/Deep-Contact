import time
import numpy as np

from Box2D import (b2ContactListener)
import cv2
import matplotlib.pyplot as plt

from ..opencv_draw import OpencvDrawFuncs

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
def run_world(world, model, timeStep, steps,
              velocityIterations, positionIterations, velocityThreshold, positionThreshold,
              iterations=False, convergenceRates=False, quiet=True):
    # We store the performance data in a dictionary
    result = {}

    # ----- Setup -----
    # Enable/disable convergence rates
    world.convergenceRates   = convergenceRates

    # Set iteration thresholds
    world.velocityThreshold = velocityThreshold
    world.positionThreshold = positionThreshold

    # Create and attach listener
    world.contactListener = WarmStartListener(model)

    # difine a drawer
    drawer = OpencvDrawFuncs(w=640, h=640, ppm=10)
    drawer.install()

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
        model.Step(world, timeStep, velocityIterations, positionIterations)

        # visiulization
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

        # Count the number of contributors for each iteration
        iterations = [len(l) for l in velocityLambdaTwoNorms]
        result["iteratorCounts"] = [np.sum([l >= i for l in iterations]) for i in range(max(iterations))]

    # Return results
    return result
