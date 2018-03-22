import numpy as np
from Box2D import b2Vec2, b2World, b2_dynamicBody
from gen_world import new_confined_clustered_circles_world
from sim_types import SimData
from scipy import spatial

if __name__ == "__main__":
    xlow, xhi = -20, 20
    ylow, yhi = 0, 40
    n_circles = 5
    sigma_coef = 1.3
    world = b2World(doSleep=False)
    world.userData=SimData("sim2grid")
    new_confined_clustered_circles_world(world, n_circles,
                                         p_ll=b2Vec2(xlow,ylow),
                                         p_hr=b2Vec2(xhi,yhi),
                                         radius_range=(1,1), sigma=sigma_coef,
                                         seed=42)
    h=3
    Pxy = np.asarray([b.position for b in world.bodies if b.type is b2_dynamicBody])
    # pX, pY = Pxy[:, 0], Pxy[:, 1]
    X, Y = np.mgrid[xlow:xhi, ylow:yhi]
    P_grid = np.c_[X.ravel(), Y.ravel()]
    KDTree = spatial.cKDTree(Pxy)
    #nn contains all neighbors within range h for every grid point
    nn = KDTree.query_ball_point(P_grid, h)
    for nn_i in range(nn.shape[0]):
        if len(nn[nn_i])>0:
            print(nn_i,"grid coordinates: ", P_grid[nn_i] ,"neighboring body indices (in Pxy not world id): ", nn[nn_i],
                    "\n neighbor(s) coordinates: ",Pxy[nn[nn_i]])