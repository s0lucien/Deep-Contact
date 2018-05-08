import numpy as np
import pandas as pd
import networkx as nx

from scipy import spatial, interpolate
from xml.etree.ElementTree import Element

from Box2D import b2World, b2_dynamicBody
from .kernel import W_poly6_2D


# Creates a dataframe with all bodies and their values given a b2World
def world_body_dataframe(world:b2World):
    bs = [[b.userData.id,
           b.position.x,
           b.position.y,
           b.mass,
           b.linearVelocity.x,
           b.linearVelocity.y,
           b.angle,
           b.angularVelocity
    ] for b in world.bodies if b.type is b2_dynamicBody]

    df = pd.DataFrame(data=bs, columns=["id", "px", "py", "mass", "vx", "vy", "theta", "omega"])
    df.id = df.id.astype(int)
    df = df.set_index("id")

    return df

# Creates a dataframe with all bodies and their values given an xml tree representing a world
def xml_body_dataframe(world:Element):
    bodies = world.find("bodies").findall("body")
    bs = [[int(b.get("index")),
           float(b.find("position").get("x")),
           float(b.find("position").get("y")),
           float(b.find("mass").get("value")),
           float(b.find("velocity").get("vx")),
           float(b.find("velocity").get("vy")),
           float(b.find("angle").get("theta")),
           float(b.find("angular_velocity").get("omega"))
    ] for b in bodies if b.get("type") == "free"]

    df = pd.DataFrame(data=bs, columns=["id", "px", "py", "mass", "vx", "vy", "theta", "omega"])
    df.id = df.id.astype(int)
    df = df.set_index("id")

    return df


# Creates a dataframe with all contacts and their values given a b2World
def world_contact_dataframe(world:b2World):
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
            normal_impulse = manifold_point.normalImpulse    # Wrong impulse, from previous step, only used for warmstarting
            tangent_impulse = manifold_point.tangentImpulse  # Wrong impulse, from previous step, only used for warmstarting

            master = c.fixtureA.body.userData.id
            slave = c.fixtureB.body.userData.id
            assert master != slave

            cs.append([master, slave, px, py, nx, ny, normal_impulse, tangent_impulse])

    df = pd.DataFrame(data=cs, columns=["master", "slave", "px", "py", "nx", "ny", "ni", "ti"])
    df.master = df.master.astype(int)
    df.slave = df.slave.astype(int)

    return df

# Creates a dataframe with all contacts and their values given an xml tree representing a world
def xml_contact_dataframe(world:Element):
    contacts = world.find("contacts").findall("contact")
    cs = [[int(c.get("master")),
           int(c.get("slave")),
           float(c.find("position").get("x")),
           float(c.find("position").get("y")),
           float(c.find("normal").get("nx")),
           float(c.find("normal").get("ny")),
           float(c.find("impulse").get("ni")),
           float(c.find("impulse").get("ti"))
    ] for c in contacts]

    df = pd.DataFrame(data=cs, columns=["master", "slave", "px", "py", "nx", "ny", "ni", "ti"])
    df.master = df.master.astype(int)
    df.slave = df.slave.astype(int)

    return df



# ----- Original, slow -----
# Creates grid of coefficients - not used because slow
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


# Creates grid of values given data and grid of coefficients - not used because slow
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


# Creates an manages grids using above functions - not used because slow
class SPHGridWorld:
    def __init__(self, world:b2World, p_ll, p_hr, xRes, yRes, h):
        self.world = world
        self.h = h

        xlo, ylo = p_ll
        xhi, yhi = p_hr
        self.x = np.arange(xlo, xhi, xRes)
        self.y = np.arange(ylo, yhi, yRes)
        self.X, self.Y =  np.mgrid[xlo:xhi:xRes, ylo:yhi:yRes]

        self.grids = {}
        self.f_interp = {}


    # The user can specify a list of "channels" to calculate grids and interpolation for, in case not all are needed
    def Step(self, channels=[]):
        self.grids = {}
        self.f_interp = {}

        df_b = world_body_dataframe(self.world)
        W_bodies = Wgrid(self.X, self.Y, df_b.px, df_b.py, df_b.index.values, self.h, f_krn=W_poly6_2D)
        b_channels = [b for b in df_b.columns.tolist() if b not in ["px", "py"]]
        if channels:
            b_channels = list(set(b_channels).intersection(channels))
        for b in b_channels:
            self.grids[b] = W_value(W_bodies, df_b, b)

        df_c = world_contact_dataframe(self.world)
        W_contacts = Wgrid(self.X, self.Y, df_c.px, df_c.py, df_c.index.values, self.h, f_krn=W_poly6_2D)
        c_channels = [c for c in df_c.columns.tolist() if c not in ["px", "py"]]
        if channels:
            c_channels = list(set(c_channels).intersection(channels))
        for c in df_c.columns.tolist():
            self.grids[c] = W_value(W_contacts, df_c, c)

        for chan in self.grids.keys():
            self.f_interp[chan] = interpolate.RectBivariateSpline(self.x, self.y, self.grids[chan])

    # Only intended to be used to query for a single point at a time
    def query(self, Px, Py, channel):
        return self.f_interp[channel](Px,Py)[0][0]



# ----- Modified, faster -----
# Creates a grid of values for each set of values - faster than using the two above functions
def create_grids(X, Y, Px, Py, values, h, f_krn=W_poly6_2D):
    '''
    splatters the points and their values onto grids, one grid per value
    :param X: grid Xs
    :param Y: grid Ys
    :param Px: Points X axis
    :param Py: Points Y axis
    :param values: point values to use
    :param h: support radius
    :param f_krn: the SPH kernel to use
    :return:
    '''
    # sanity check
    # assert Px.shape[1] == Py.shape[1] == 1
    # assert Px.shape[0] == Py.shape[0]
    # assert X.shape == Y.shape
    # assert values.shape[0] = Px.shape[0]
    n = np.shape(values)[1]
    if n == 0:
        return []

    Xsz, Ysz = X.shape
    grids = [np.zeros((Xsz, Ysz), dtype=float) for _ in range(n)]  # TODO: change to sparse

    if len(values) > 0:
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

                # For all neighbors, multiply W with all values and store in V
                V = [.0]*n
                for nni in range(len(g_nn)):
                    vs = values[g_nn[nni]]
                    for i in range(n):
                        V[i] += W[nni] * vs[i]

                # Store V in grids
                for i in range(n):
                    grids[i][xi, yi] = V[i]

    return grids


# Creates and manages grids given dataframes with body and contact values
class SPHGridManager:
    def __init__(self, p_ll, p_ur, xRes, yRes, h):
        xlo, ylo = p_ll
        xhi, yhi = p_ur

        self.h = h
        self.x = np.arange(xlo, xhi, xRes)
        self.y = np.arange(ylo, yhi, yRes)
        self.X, self.Y =  np.mgrid[xlo:xhi:xRes, ylo:yhi:yRes]

        self.grids = {}
        self.f_interp = {}

    def reset(self):
        self.grids = {}
        self.f_interp = {}

    # The user specifies a list of "channels" to calculate grids for
    def create_grids(self, df, channels):
        df_channels = [d for d in df.columns.tolist() if d not in ["px", "py"]]
        known = []
        for c in channels:
            if c in df_channels:
                known.append(c)
            else:
                print("Unknown channel: ", c)
                logging.info("Unknown channel: " + c)

        channels = known
        if not channels:
            return

        data = df[channels].values
        grids = create_grids(self.X, self.Y, df.px, df.py, data, self.h, f_krn=W_poly6_2D)
        for i in range(len(channels)):
            self.grids[channels[i]] = grids[i]

    # The user specifies a list of "channels" to calculate interpolation for
    def create_interp(self, channels=[]):
        for c in channels:
            grid = self.grids.get(c)
            if grid is not None:
                self.f_interp[c] = interpolate.RectBivariateSpline(self.x, self.y, grid)
            else:
                print("Unknown channel: ", c)
                logging.info("Unknown channel: " + c)

    # Only intended to be used to query for a single point at a time
    def query(self, Px, Py, channel):
        return self.f_interp[channel](Px,Py)[0][0]



# Creates a contact graph?
def contact_graph(world: b2World):
    df = contact_properties(world)
    G = nx.MultiDiGraph()
    for i, row in df.iterrows():
        e = G.add_edge(row.master, row.slave,
                       attr_dict={"px": row.px, "py": row.py, "nx": row.nx, "ny": row.ny,
                                  "normal_impulse": row.ni, "tangent_impulse": row.ti})
    return G
