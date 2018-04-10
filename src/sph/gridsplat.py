import numpy as np
from Box2D import b2World, b2_dynamicBody
from scipy import spatial
from .kernel import W_poly6_2D
import pandas as pd
import networkx as nx
from scipy import interpolate


def W_value(W_grid, data, channel_name):
    Xsz, Ysz = W_grid.shape
    Z = np.zeros(W_grid.shape)
    for i in range(Xsz):
        for j in range(Ysz):
            if W_grid[i,j] != 0:
                z = 0
                for (id, w) in W_grid[i,j]:
                    z += data.loc[id][channel_name] * w
                Z[i,j] = z
    return Z


def Wgrid(X, Y, Px, Py, ID, h, f_krn=W_poly6_2D):
    '''
    splatters the points onto a grid , resulting in coefficients for every point
    :param X: grid X
    :param Y: grid Y
    :param Px : Points X axis
    :param Py: Points Y axis
    :param id: how are points identified
    :param h: support radius
    :param f_krn: the SPH kernel to use
    :return:
    '''
    # sanity check
    # assert Px.shape[1] == Py.shape[1] == 1
    # assert Px.shape[0] == Py.shape[0]
    # assert X.shape == Y.shape
    Xsz, Ysz = X.shape
    W_grid = np.zeros((Xsz, Ysz), dtype=object)  # TODO: change to sparse

    if len(ID) > 0:
        P_grid = np.c_[X.ravel(), Y.ravel()]
        Pxy = np.c_[Px, Py]  # stack the points as row-vectors

        KDTree = spatial.cKDTree(Pxy)
        # nn contains all neighbors within range h for every grid point
        NN = KDTree.query_ball_point(P_grid, h)
        for i in range(NN.shape[0]):
            if len(NN[i]) > 0:
                xi, yi = np.unravel_index(i, (Xsz, Ysz))
                g_nn = NN[i]  # grid nearest neighbors
                r = P_grid[i] - Pxy[g_nn, :]  # the 3rd column is the body id
                W = f_krn(r.T, h)
                if W_grid[xi, yi] == 0:
                    W_grid[xi, yi] = []

                Ws = []
                for nni in range(len(g_nn)):
                    body_id = ID[g_nn[nni]]
                    tup = (body_id, W[nni])  # we store the values as tuples (body_id, W) at each grid point
                    Ws.append(tup)
                W_grid[xi, yi] += Ws  # to merge the 2 lists we don't use append

    return W_grid


def body_properties(world: b2World):
    bs = [[b.userData.id,
           b.position.x,
           b.position.y, # do we need positions or just the values?
           b.mass,
           b.linearVelocity.x,
           b.linearVelocity.y,
           b.inertia,
           b.angle,
           b.angularVelocity
    ] for b in world.bodies if b.type is b2_dynamicBody]

    df = pd.DataFrame(data=bs, columns=["id", "px","py", "mass", "vx", "vy", "inertia", "angle", "spin"])
    df.id = df.id.astype(int)
    df = df.set_index("id")

    return df


def contact_properties(world: b2World):
    cs = []
    for i in range(world.contactCount):
        c = world.contacts[i]
        if not c.touching:
            continue

        for ii in range(c.manifold.pointCount):
            world_point = c.worldManifold.points[ii]
            px = world_point[0]
            py = world_point[1]
            normal = c.worldManifold.normal
            nx = normal[0]
            ny = normal[1]

            manifold_point = c.manifold.points[ii]
            normal_impulse = manifold_point.normalImpulse
            tangent_impulse = manifold_point.tangentImpulse

            master = c.fixtureA.body.userData.id
            slave = c.fixtureB.body.userData.id
            assert master != slave

            cs.append([master, slave, px, py, nx, ny, normal_impulse, tangent_impulse])

    df = pd.DataFrame(data=cs, columns=["master", "slave", "px", "py", "nx", "ny", "normal_impulse", "tangent_impulse"])
    df.master = df.master.astype(int)
    df.slave = df.slave.astype(int)
    df = df.set_index(["master", "slave"])

    return df


def contact_graph(world: b2World):
    df = contact_properties(world)
    G = nx.MultiDiGraph()
    for i, row in df.iterrows():
        e = G.add_edge(i[0], i[1],
                       attr_dict={"px": row.px, "py": row.py, "nx": row.nx, "ny": row.ny,
                                  "normal_impulse": row.normal_impulse, "tangent_impulse": row.tangent_impulse})
    return G


class SPHGridWorld:
    def __init__(self, world:b2World, p_ll, p_hr, xRes, yRes, h):
        self.world = world
        self.h = h

        xlo, ylo = p_ll
        xhi, yhi = p_hr
        self.X, self.Y =  np.mgrid[xlo:xhi:xRes, ylo:yhi:yRes]

        self.grids = {}
        self.f_interp = {}


    # The user can specify a list of "channels" to calculate grids and interpolation for, in case not all are needed
    def Step(self, channels=[]):
        self.grids = {}
        self.f_interp = {}

        df_b = body_properties(self.world)
        W_bodies = Wgrid(self.X, self.Y, df_b.px, df_b.py, df_b.index.values, self.h, f_krn=W_poly6_2D)
        b_channels = [b for b in df_b.columns.tolist() if b not in ["px", "py"]]
        if channels:
            b_channels = set(b_channels).intersection(channels)
        for b in b_channels:
            self.grids[b] = W_value(W_bodies, df_b, b)

        df_c = contact_properties(self.world)
        W_contacts = Wgrid(self.X, self.Y, df_c.px, df_c.py, df_c.index.values, self.h, f_krn=W_poly6_2D)
        c_channels = [c for c in df_c.columns.tolist() if c not in ["px", "py"]]
        if channels:
            c_channels = set(c_channels).intersection(channels)
        for c in df_c.columns.tolist():
            self.grids[c] = W_value(W_contacts, df_c, c)

        for chan in self.grids.keys():
            self.f_interp[chan] = interpolate.interp2d(self.X, self.Y, self.grids[chan], kind="linear")

    def query(self, Px, Py, channel):
        return self.f_interp[channel](Px,Py)[0]
