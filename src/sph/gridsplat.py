import numpy as np
import pandas as pd
import networkx as nx
import logging

from scipy import spatial, interpolate
from xml.etree.ElementTree import Element

from Box2D import b2World, b2_dynamicBody
from .kernel import W_poly6_2D

logging.basicConfig(level=logging.INFO)


# Creates a dataframe with all bodies and their values given a b2World
def world_body_dataframe(world:b2World):
    bs = [[b.userData.id,
           b.position.x,
           b.position.y,
           b.mass,
           b.inertia,
           b.linearVelocity.x,
           b.linearVelocity.y,
           b.angle,
           b.angularVelocity
    ] for b in world.bodies if b.type is b2_dynamicBody]

    df = pd.DataFrame(data=bs, columns=["id", "px", "py", "mass", "inertia", "vx", "vy", "theta", "omega"])
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
           float(b.find("inertia").get("value")),
           float(b.find("velocity").get("vx")),
           float(b.find("velocity").get("vy")),
           float(b.find("angle").get("theta")),
           float(b.find("angular_velocity").get("omega"))
    ] for b in bodies if b.get("type") == "free"]

    df = pd.DataFrame(data=bs, columns=["id", "px", "py", "mass", "inertia", "vx", "vy", "theta", "omega"])
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


# Creates a set of grids of values using SPH
def create_grids(Gx, Gy, Px, Py, values, h, f_krn=W_poly6_2D):
    '''
    splatters the points and their values onto grids, one grid per value
    :param Gx:     Grid Xs
    :param Gy:     Grid Ys
    :param Px:     Point Xs
    :param Py:     Point Ys
    :param values: Point values
    :param h:      Support radius
    :param f_krn:  The SPH kernel to use
    :return:
    '''
    n = np.shape(values)[1]
    if n == 0:
        return []

    # We create the grid
    Gx_sz, Gy_sz = Gx.shape
    grids = np.zeros((n, Gx_sz, Gy_sz), dtype=float) # TODO: change to sparse

    # If there are no values, return empty grid
    if np.size(values) == 0:
        return grids

    # Create array where each row is a x- and y-coordinate of a node in the grid
    P_grid   = np.c_[Gx.ravel(), Gy.ravel()]
    P_points = np.c_[Px, Py]

    # For each point we determine all grid nodes within radius h
    KDTree = spatial.cKDTree(P_grid)
    NNs = KDTree.query_ball_point(P_points, h)

    # For each point
    for i in range(NNs.shape[0]):
        # We determine distances between point and neighbouring grid nodes
        neighbours = NNs[i]
        rs = P_points[i] - P_grid[neighbours]

        # We determine weights for each neighbouring grid node
        Ws = f_krn(rs.T, h)

        # For all neighbors, multiply weight with all point values and store in grid
        for j in range(len(neighbours)):
            gxi, gyi = np.unravel_index(neighbours[j], (Gx_sz, Gy_sz))
            for k in range(n):
                grids[k, gxi, gyi] += Ws[j] * values[i, k]

    return grids


# Creates and manages grids
class SPHGridManager:
    def __init__(self, p_ll, p_ur, xRes, yRes, h):
        xlo, ylo = p_ll
        xhi, yhi = p_ur

        self.h = h
        self.x = np.arange(xlo, xhi+xRes, xRes)
        self.y = np.arange(ylo, yhi+xRes, yRes)
        self.X, self.Y = np.mgrid[xlo:(xhi+xRes):xRes, ylo:(yhi+yRes):yRes]

        self.reset()

    def reset(self):
        self.grids = {}
        self.f_interp = {}
        self.trees = {}

    # Adds a grid to the grid manager
    def add_grid(self, grid, channel):
        self.grids[channel] = grid

    # The user specifies a list of "channels" to calculate grids for, and provides the data
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
    def create_interp(self, channels):
        for c in channels:
            grid = self.grids.get(c)
            if grid is not None:
                self.f_interp[c] = interpolate.RectBivariateSpline(self.x, self.y, grid)
            else:
                logging.info("Unknown channel: " + c)

    # Only intended to be used to query for a single point at a time
    def query_interp(self, Px, Py, channel):
        return self.f_interp[channel](Px,Py)[0][0]


    # Takes a list of particles and a list of grids as input, and returns a list
    # of lists of values, one list for each particle with one value for each grid
    def grids_to_particles(self, grids, points):
        grid_nodes = np.c_[self.X.ravel(), self.Y.ravel()]
        N = grid_nodes.shape[0]

        # We create the tree
        tree = spatial.cKDTree(points)

        # For each grid point, determine neighbouring particles
        NNs = tree.query_ball_point(grid_nodes, self.h)

        # An array for storing a value for each grid for each particle
        values = np.zeros([points.shape[0], grids.shape[0]], dtype=np.float64)

        # For each grid node
        for i in range(N):
            # We determine distances between grid node and neighbouring particles
            neighbours = NNs[i]
            if len(neighbours) > 0:
                rs = grid_nodes[i] - points[neighbours]

                # We determine weights for each neighbouring particle
                Ws = W_poly6_2D(rs.T, self.h)

                # iy and ix are the index into the unflattened array
                iy, ix = np.unravel_index(i, self.X.shape)

                # We multiply weights with values to get the particle values
                for j in range(len(neighbours)):
                    n = neighbours[j]
                    w = Ws[j]
                    for k in range(grids.shape[0]):
                        values[n, k] += w * grids[k, iy, ix]

        return values


# Creates a contact graph?
def contact_graph(world: b2World):
    df = contact_properties(world)
    G = nx.MultiDiGraph()
    for i, row in df.iterrows():
        e = G.add_edge(row.master, row.slave,
                       attr_dict={"px": row.px, "py": row.py, "nx": row.nx, "ny": row.ny,
                                  "normal_impulse": row.ni, "tangent_impulse": row.ti})
    return G
