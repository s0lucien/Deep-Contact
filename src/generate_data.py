from .warm_starting.warm_start import run_world
from .warm_starting.identity_grid_model import IdentityGridModel
from .gen_world import new_confined_clustered_circles_world

from Box2D import (b2World, b2Vec2)


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-e', '--export-path', dest='export_path')
    parser.add_option('-s', '--steps', dest='steps', default='1000')
    parser.add_option('-v', '--visualize', action='store_true', dest='visualize')

    options, _ = parser.parse_args()

    # Let's begin with create one world
    world = b2World()

    # setting with the lovely world
    nBodies = 100
    seed = 1234
    sigma_coef = 1.2
    xlow, xhi = 0, 50
    ylow, yhi = 0, 50
    r = (1, 1)

    # Let our world filled with objects
    new_confined_clustered_circles_world(
        world, nBodies, b2Vec2(xlow, ylow), b2Vec2(xhi, yhi), r, sigma_coef, seed,
    )

    # Grid parameters
    p_ll = (xlow, ylow)
    p_ur = (xhi, yhi)

    xRes = 0.75
    yRes = 0.75

    h = 1

    model = IdentityGridModel(world, p_ll, p_ur, xRes, yRes, h)

    # Timestep
    timeStep = 1.0 / 100
    # Iteration limits
    velocityIterations = 5000
    positionIterations = 2500
    # Iteration thresholds
    velocityThreshold = 6*10**-5
    positionThreshold = 2*10**-5
    # Number of steps
    steps = int(options.steps)

    result = run_world(
        world,
        timeStep,
        steps,
        velocityIterations,
        positionIterations,
        velocityThreshold=velocityThreshold,
        positionThreshold=positionThreshold,
        model=model,
        iterations=True,
        convergenceRates=True,
        quiet=False,
        visualize=options.visualize,
        export_path=options.export_path,
    )
