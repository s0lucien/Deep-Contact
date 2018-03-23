import numpy as np
from Box2D import b2Vec2, b2World, b2_dynamicBody
from gen_world import new_confined_clustered_circles_world
from sim_types import SimData
from scipy import spatial
import matplotlib.pyplot as plt

from sph.kernel import W_poly6_2D

if __name__ == "__main__":
    #uncomment to get the seed of a specific (working) configuration
    # np.random.seed(None);st0 = np.random.get_state();print(st0);np.random.set_state(st0)
    xlow, xhi = -5, 5
    ylow, yhi = 0, 10
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
    xspace, yspace= 1,1
    Pxy = np.asarray([[b.position.x, b.position.y, b.userData.id] for b in world.bodies if b.type is b2_dynamicBody])
    # pX, pY = Pxy[:, 0], Pxy[:, 1]
    X, Y = np.mgrid[xlow:xhi:xspace, ylow:yhi:xspace]
    Xsz,Ysz = X.shape
    P_grid = np.c_[X.ravel(), Y.ravel()]
    KDTree = spatial.cKDTree(Pxy[:,0:2])
    #nn contains all neighbors within range h for every grid point
    NN = KDTree.query_ball_point(P_grid, h)
    W_grid = np.zeros((Xsz,Ysz),dtype=object) #TODO: change to sparse
    for i in range(NN.shape[0]):
        if len(NN[i])>0:
            xi , yi = np.unravel_index(i,(Xsz,Ysz))
            g_nn = NN[i] # grid nearest neighbors
            r = P_grid[i] - Pxy[g_nn,0:2] # the 3rd column is the body id
            W = W_poly6_2D(r.T,h)
            if W_grid[xi,yi] == 0:
                W_grid[xi, yi] = []
            Ws=[]
            for nni in range(len(g_nn)):
                body_id = int(Pxy[g_nn[nni],2])
                tup = (body_id,W[nni])  # we store the values as tuples (body_id, W) at each grid point
                Ws.append(tup)
            W_grid[xi, yi] += Ws # to merge the 2 lists we don't use append

# visualize the sparsity pattern
Wspy = np.copy(W_grid)
for i in range(Xsz):
    for j in range(Ysz):
        if Wspy[i,j] != 0:
            Wspy[i,j] =  len(Wspy[i,j])
plt.spy(Wspy)
plt.show()