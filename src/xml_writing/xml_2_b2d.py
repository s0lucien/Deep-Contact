from sim_types import BodyData, SimData, dcCircleShape, dcLoopShape, GenWorld
import xml.etree.ElementTree as ET
from Box2D import b2World, b2Vec2
import os


def xml_2_b2bodies (world:b2World, xmlStr):
    tree = ET.ElementTree(ET.fromstring(xmlStr))
    for body in tree.getroot().findall("body"):
        id = int(body.attrib["index"])
        type = body.attrib["type"]
        p = body.find("position")
        px, py = float(p.attrib['x']), float(p.attrib['y'])
        v = body.find("velocity")
        vx, vy = float(v.attrib['vx']), float(v.attrib['vy'])
        shape = body.find("shape").attrib['value']
        spin = float(body.find("spin").attrib['omega'])
        mass = float(body.find("mass").attrib['value'])
        orientation = float(body.find("orientation").attrib['theta'])
        inertia = float(body.find("inertia").attrib['value'])

        if type == "free":
            bod = world.CreateDynamicBody(
                position=b2Vec2(px, py),
                fixtures=eval(shape).fixture,
            )
        elif type == "fixed":
            bod = world.CreateStaticBody(
                position=b2Vec2(px, py),
                fixtures=eval(shape).fixture,
            )

        bod.userData = BodyData(b_id=id, shape=shape)
        bod.mass = mass
        bod.inertia = inertia
        bod.linearVelocity = b2Vec2(vx, vy)
        bod.angle = orientation
        bod.angularVelocity = spin


def xml_2_b2world (xmlStr):
    tree = ET.ElementTree(ET.fromstring(xmlStr))
    cfg = tree.getroot()
    name = cfg.attrib['name']
    time = float(cfg.attrib['time'])
    dt = float(cfg.attrib['dt'])
    return name,time,dt


class XMLImporter:
    def __init__(self, import_file, world=None):
        # hacky way of allowing xmlStr to be passed as arg
        try:
            with open(import_file, 'r') as content_file:
                self.xmlStr = content_file.read()
                self.file = os.path.basename(import_file)
                self.file = os.path.splitext(self.file)
                # we save the filename with the epoch after the final underscore
                f = self.file[0]
                self.epoch = int(f[f.rfind("_") + 1:])  # rfind is lastIndexOf, [i:] slice str after i
                self.cfgname = f[:f.rfind("_")]
        except FileNotFoundError:
            self.xmlStr = import_file
            # epoch cannot be determined from content
            self.epoch = None
        if world is None:
            self.world = b2World()
            self.world.userData = SimData()
        else:
            self.world = world
        # set the default options that we set to each world
        GenWorld(self.world)
        self.world.userData.step = self.epoch
        n,t,dt = xml_2_b2world(self.xmlStr)
        self.world.userData.name = n
        self.world.userData.sim_t = t
        self.world.userData.dt = dt

    def load(self):
        xml_2_b2bodies(self.world, self.xmlStr)
        self.world.initialized = True
