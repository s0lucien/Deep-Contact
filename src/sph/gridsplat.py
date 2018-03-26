import numpy as np
from Box2D import b2World, b2_dynamicBody
from scipy import spatial
from .kernel import W_poly6_2D
import pandas as pd


def W_grid_poly6(world: b2World, h, p_ll, p_hr, xRes, yRes):
    '''
    splatters the points onto a grid , resulting in coefficients for every point
    :param world: b2world that contins SimData information
    :param h: support radius
    :param p_ll: lower left point of the grid
    :param p_hr: upper right point of the grid
    :param xRes: resolution on the horizontal axis
    :param yRes: resolution on the vertical axis
    :return:
    '''
    xlow, ylow = p_ll
    xhi, yhi = p_hr
    Pxy = np.asarray([[b.position.x, b.position.y, b.userData.id] for b in world.bodies if b.type is b2_dynamicBody])
    # pX, pY = Pxy[:, 0], Pxy[:, 1]
    X, Y = np.mgrid[xlow:xhi:xRes, ylow:yhi:yRes]
    Xsz, Ysz = X.shape
    P_grid = np.c_[X.ravel(), Y.ravel()]
    KDTree = spatial.cKDTree(Pxy[:, 0:2])
    # nn contains all neighbors within range h for every grid point
    NN = KDTree.query_ball_point(P_grid, h)
    W_grid = np.zeros((Xsz, Ysz), dtype=object)  # TODO: change to sparse
    for i in range(NN.shape[0]):
        if len(NN[i]) > 0:
            xi, yi = np.unravel_index(i, (Xsz, Ysz))
            g_nn = NN[i]  # grid nearest neighbors
            r = P_grid[i] - Pxy[g_nn, 0:2]  # the 3rd column is the body id
            W = W_poly6_2D(r.T, h)
            if W_grid[xi, yi] == 0:
                W_grid[xi, yi] = []
            Ws = []
            for nni in range(len(g_nn)):
                body_id = int(Pxy[g_nn[nni], 2])
                tup = (body_id, W[nni])  # we store the values as tuples (body_id, W) at each grid point
                Ws.append(tup)
            W_grid[xi, yi] += Ws  # to merge the 2 lists we don't use append
    return W_grid


def body_properties(world: b2World):
    B = np.asarray([[b.userData.id,
                     # b.position.x,
                     # b.position.y, # do we need positions or just the values?
                     b.mass,
                     b.linearVelocity.x,
                     b.linearVelocity.y,
                     b.inertia,
                     b.angle,
                     b.angularVelocity
                     ] for b in world.bodies if b.type is b2_dynamicBody])

    df = pd.DataFrame(data=B, columns=["id",
                                       # "px","py",
                                       "mass", "vx", "vy", "inertia", "angle", "spin"])
    df.id = df.id.astype(int)
    df = df.set_index("id")
    return df


def contact_properties(world: b2World):
    cs = []
    for i in range(world.contactCount):
        c = world.contacts[i]
        for ii in range(c.manifold.pointCount):
            point = c.worldManifold.points[ii]
            manifold_point = c.manifold.points[ii]
            normal = c.worldManifold.normal
            normal_impulse = manifold_point.normalImpulse
            tangent_impulse = manifold_point.tangentImpulse
            master = c.fixtureA.body.userData.id
            slave = c.fixtureB.body.userData.id
            px = point[0]
            py = point[1]
            nx = normal[0]
            ny = normal[1]
            assert master != slave
            cs.append([master, slave, px, py, nx, ny, normal_impulse, tangent_impulse])
    C = np.asarray(cs)

    if C.size == 0:
        raise ValueError("Contacts should not be empty !!")
    df = pd.DataFrame(data=C, columns=["master", "slave", "px", "py", "nx", "ny", "normal_impulse", "tangent_impulse"])
    # perform some formatting on the columns
    df.master = df.master.astype(int)
    df.slave = df.slave.astype(int)
    # df = df.set_index("master")  # TODO: change to composite
    return df
