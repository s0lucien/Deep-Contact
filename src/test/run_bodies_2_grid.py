import numpy as np
from Box2D import b2Vec2, b2World, b2_dynamicBody
from gen_world import new_confined_clustered_circles_world
from sim_types import SimData
from sph.gridsplat import Wgrid , W_value, body_properties
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #uncomment to get the seed of a specific (working) configuration
    # np.random.seed(None);st0 = np.random.get_state();print(st0);np.random.set_state(st0)
    xlow, xhi = -5, 2
    ylow, yhi = 0, 15
    n_circles = 3
    sigma_coef = 1.3
    world = b2World(doSleep=False)
    world.userData=SimData("sim2grid")
    new_confined_clustered_circles_world(world, n_circles,
                                         p_ll=b2Vec2(xlow,ylow),
                                         p_hr=b2Vec2(xhi,yhi),
                                         radius_range=(1,1), sigma=sigma_coef,
                                         seed=None)
    h=3
    xRes, yRes= 1,1
    Pxy = np.asarray([[b.position.x, b.position.y, b.userData.id] for b in world.bodies if b.type is b2_dynamicBody])
    Px, Py,ID = Pxy[:, 0][:,np.newaxis], Pxy[:, 1][:,np.newaxis], Pxy[:,2]
    X, Y = np.mgrid[xlow:xhi:xRes, ylow:yhi:yRes]
    df = body_properties(world)

    W_grid = Wgrid(X,Y,Px,Py,ID,h)

    W = W_value(W_grid, df, "mass")
    assert W.shape == W_grid.shape
    plt.spy(W)
    plt.show()

