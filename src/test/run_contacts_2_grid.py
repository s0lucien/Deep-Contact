import numpy as np
from Box2D import b2Vec2, b2World, b2_dynamicBody
from gen_world import new_confined_clustered_circles_world
from sim_types import SimData
from sph.gridsplat import Wgrid
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)
from sph.gridsplat import contact_properties


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
    while world.contactCount <5 :
        world.Step(0.01, 100, 100)
        logging.debug("stepped 0.1")
    h=3
    xRes, yRes= 1,1
    df = contact_properties(world)

    Px, Py,ID = df.px[:,np.newaxis], df.py[:,np.newaxis], list(zip(df.master,df.slave))
    X, Y = np.mgrid[xlow:xhi:xRes, ylow:yhi:yRes]

    W_grid = Wgrid(X,Y,Px,Py,ID,h)

# visualize the sparsity pattern
Wspy = np.copy(W_grid)
Xsz, Ysz = Wspy.shape
for i in range(Xsz):
    for j in range(Ysz):
        if Wspy[i,j] != 0:
            Wspy[i,j] = len(Wspy[i,j])
plt.spy(Wspy)
plt.show()
