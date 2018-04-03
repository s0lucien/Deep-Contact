import numpy as np
from Box2D import b2Vec2, b2World, b2_dynamicBody
from ..gen_world import new_confined_clustered_circles_world
from ..sim_types import SimData
from ..sph.gridsplat import C_grid_poly6
from ..sph.kernel import W_poly6_2D
from .run_confined import Confined
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
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

    while len(world.contacts) == 0:
        world.Step(0.01, 100, 100)
        print(len(world.contacts))
    h=3
    xspace, yspace= 1,1
    g_dict, C_grid = C_grid_poly6(world,h,(xlow,ylow),(xhi,yhi), xspace, yspace)

    # visualize the sparsity pattern
    Cspy = np.copy(C_grid)
    Xsz, Ysz = Cspy.shape
    for i in range(Xsz):
        for j in range(Ysz):
            if Cspy[i,j] != 0:
                Cspy[i,j] = len(Wspy[i,j])
    plt.spy(Cspy)
    plt.show()
