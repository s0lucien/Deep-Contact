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
                for (body_id,w) in W_grid[i,j]:
                    z += data.loc[body_id][channel_name]*w
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
    P_grid = np.c_[X.ravel(), Y.ravel()]
    Pxy = np.c_[Px, Py]  # stack the points as row-vectors
    KDTree = spatial.cKDTree(Pxy)
    # nn contains all neighbors within range h for every grid point
    NN = KDTree.query_ball_point(P_grid, h)
    W_grid = np.zeros((Xsz, Ysz), dtype=object)  # TODO: change to sparse
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
    B = np.asarray([[b.userData.id,
                     b.position.x,
                     b.position.y, # do we need positions or just the values?
                     b.mass,
                     b.linearVelocity.x,
                     b.linearVelocity.y,
                     b.inertia,
                     b.angle,
                     b.angularVelocity
                     ] for b in world.bodies if b.type is b2_dynamicBody])

    df = pd.DataFrame(data=B, columns=["id",
                                       "px","py",
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
        return pd.DataFrame(columns=["master", "slave", "px", "py", "nx", "ny", "normal_impulse", "tangent_impulse","e"])
        # raise ValueError("Contacts should not be empty !!")
    df = pd.DataFrame(data=C, columns=["master", "slave", "px", "py", "nx", "ny", "normal_impulse", "tangent_impulse"])
    # perform some formatting on the columns
    df.master = df.master.astype(int)
    df.slave = df.slave.astype(int)
    edges = [tuple(row[col] for col in ['master', 'slave']) for _, row in df.iterrows()]
    e_ix = pd.MultiIndex.from_tuples(edges, names=["master","slave"])
    #a = pd.concat([df, edges], axis=1)
    df = df.set_index(e_ix)
    return df


def contact_graph(world: b2World):
    df = contact_properties(world)
    G = nx.MultiDiGraph()
    for i, row in df.iterrows():
        e = G.add_edge(row.master, row.slave,
                       attr_dict={"px": row.px, "py": row.py, "nx": row.nx, "ny": row.ny,
                                  "normal_impulse": row.normal_impulse, "tangent_impulse": row.tangent_impulse})
    return G


class SPHGridWorld:
    def __init__(self, world:b2World, p_ll, p_hr, xRes, yRes, h):
        self.world = world
        self.h = h
        xlo, ylo = p_ll
        xhi, yhi = p_hr
        self.X,self.Y =  np.mgrid[xlo:xhi:xRes, ylo:yhi:yRes]
        self.grids = {}
        self.f_interp = {}


    def Step(self):
        self.df_b = body_properties(self.world)
        self.df_c = contact_properties(self.world)
        bPx = self.df_b.px
        bPy = self.df_b.py
        bID = self.df_b.index.values
        self.W_bodies = Wgrid(self.X, self.Y, bPx, bPy, bID, self.h, f_krn=W_poly6_2D)
        b_channels = self.df_b.columns.tolist()
        for b in b_channels:
            if b not in ["px", "py", "id"]:
                self.grids[b] = W_value(self.W_bodies, self.df_b, b)
        c_channels = self.df_c.columns.tolist()
        if not self.df_c.empty:
            cPx = self.df_c.px
            cPy = self.df_c.py
            cID = self.df_c.index.values
            self.W_contacts = Wgrid(self.X, self.Y, cPx, cPy, cID, self.h, f_krn=W_poly6_2D)
            for c in c_channels:
                if c not in ["px", "py", "e", "master", "slave"]:
                    self.grids[c] = W_value(self.W_contacts, self.df_b, c)
        else :
            for c in c_channels:
                self.grids.pop(c,None)
        for chan in self.grids.keys():
            self.f_interp[chan] = interpolate.interp2d(self.X, self.Y, self.grids[chan], kind="linear")

    def query(self, Px, Py, channel):
        return self.f_interp[channel](Px,Py)