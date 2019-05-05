import pandas as pd

from scipy import spatial, interpolate
from xml.etree.ElementTree import Element

from Box2D import b2World, b2_dynamicBody


# Creates a dataframe with all bodies, and one with all contacts,
# and their values given a b2World
def dataframes_from_b2World(world:b2World):
    # Bodies
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

    df_b = pd.DataFrame(data=bs, columns=["id", "px", "py", "mass", "inertia", "vx", "vy", "theta", "omega"])
    df_b.id = df_b.id.astype(int)
    df_b = df_b.set_index("id")

    # Contacts
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

            cs.append([master, slave, px, py, nx, ny, normal_impulse, tangent_impulse])

    df_c = pd.DataFrame(data=cs, columns=["master", "slave", "px", "py", "nx", "ny", "ni", "ti"])
    df_c.master = df_c.master.astype(int)
    df_c.slave = df_c.slave.astype(int)

    return df_b, df_c


# Creates a dataframe with all bodies, and one with all contacts,
# given an xml tree representing a world
def dataframes_from_xml(world:Element):
    # Bodies
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

    df_b = pd.DataFrame(data=bs, columns=["id", "px", "py", "mass", "inertia", "vx", "vy", "theta", "omega"])
    df_b.id = df_b.id.astype(int)
    df_b = df_b.set_index("id")

    # Contacts
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

    df_c = pd.DataFrame(data=cs, columns=["master", "slave", "px", "py", "nx", "ny", "ni", "ti"])
    df_c.master = df_c.master.astype(int)
    df_c.slave = df_c.slave.astype(int)

    return df_b, df_c
