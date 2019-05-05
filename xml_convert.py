from Box2D import (b2World, b2Body, b2Contact,b2_dynamicBody, b2Vec2)
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import os, errno
from sim_types import BodyData, SimData, dcCircleShape, dcLoopShape, GenWorld
from pathlib import Path


def body_to_xml(body: b2Body):
    body_xml = Element('body')
    body_xml.set('index', str(body.userData.id))

    if body.type is b2_dynamicBody :
        type  = 'free'
    elif body.type is 0:
        type = 'fixed'
    else :
        raise Exception("unidentified body type encountered")
    body_xml.set('type', type)

    # mass
    mass = SubElement(body_xml, 'mass')
    mass.set('value', str(body.mass))

    # inertia
    inertia = SubElement(body_xml, "inertia")
    inertia.set('value', str(body.inertia))

    # position
    pos = SubElement(body_xml, 'position')
    pos.set('x', str(body.position.x))
    pos.set('y', str(body.position.y))

    # velocity
    v = SubElement(body_xml, 'velocity')
    v.set('vx', str(body.linearVelocity.x))
    v.set('vy', str(body.linearVelocity.y))

    # orientation
    angle = SubElement(body_xml, 'angle')
    angle.set('theta', str(body.angle))

    # spin
    spin = SubElement(body_xml, 'angular_velocity')
    spin.set('omega', str(body.angularVelocity))

    # shape
    shape = SubElement(body_xml, 'shape')
    shape.set('value', str(body.userData.shape))

    return body_xml


def contact_to_xml(contact: b2Contact):
    contact_xmls = []
    for i in range(contact.manifold.pointCount):
        # index
        ct_xml = Element('contact')
        ct_xml.set('index', str(int(contact.userData) + i))
        ct_xml.set("master", str(contact.fixtureA.body.userData.id))
        ct_xml.set("slave", str(contact.fixtureB.body.userData.id))

        # position
        point = contact.worldManifold.points[i]
        point_xml = SubElement(ct_xml, 'position')
        point_xml.set('x', str(point[0]))
        point_xml.set('y', str(point[1]))

        # normal
        normal = contact.worldManifold.normal
        normal_xml = SubElement(ct_xml, "normal")
        normal_xml.set("nx", str(normal.x))
        normal_xml.set("ny", str(normal.y))

        # impulses are added later on due to reasons

        contact_xmls.append(ct_xml)

    return contact_xmls





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

class XMLExporter:
    def __init__(self, world:b2World, export_root):
        self.export_root = export_root
        self.world = world
        self.simData=world.userData

        self.reset()

    # Creates a new xml tree in preparation for a new step of the simulation
    def reset(self):
        self.cfg      = Element("configuration")
        self.bodies   = SubElement(self.cfg, "bodies")
        self.contacts = SubElement(self.cfg, "contacts")

        self.cfg.set("name", str(self.simData.name))
        self.cfg.set("time", str(self.simData.sim_t))
        self.cfg.set("dt", str(self.simData.dt))

        self.cfg.set("vel_iter", str(self.simData.vel_iter))
        self.cfg.set("pos_iter", str(self.simData.pos_iter))
        self.cfg.set("vel_thres", str(self.simData.vel_thres))
        self.cfg.set("pos_thres", str(self.simData.pos_thres))


    # Store all bodies in the world as xml
    def snapshot_bodies(self):
        for b in self.world.bodies:
            xb = body_to_xml(b)
            if xb is not None:
                self.bodies.append(xb)

    # Store all contacts in the world as xml
    def snapshot_contacts(self):
        for c in self.world.contacts:
            self.snapshot_contact(c)

    # Store the given contact as xml
    def snapshot_contact(self, contact:b2Contact):
        ct_xml = contact_to_xml(contact)
        for ct_pt in ct_xml:
            # import pdb; pdb.set_trace()
            self.contacts.append(ct_pt)

    # Add the given impulse to the already-existing xml representing the given contact
    def snapshot_impulse(self, contact:b2Contact, impulse):
        for i in range(contact.manifold.pointCount):
            # We find the corresponding contact xml
            index = str(int(contact.userData) + i)
            ct_xml_list = self.contacts.findall(".//contact[@index='" + index + "']")

            if len(ct_xml_list) != 1:
                raise AssertionError("Found more (or less) than one contact with the given id in the xml tree!")
            ct_xml = ct_xml_list[0]

            # We add the impulse
            impulse_xml = SubElement(ct_xml, "impulse")
            impulse_xml.set("ni", str(impulse.normalImpulses[i]))
            impulse_xml.set("ti", str(impulse.tangentImpulses[i]))

    # Save all the stored xml to a file
    def save_snapshot(self,filename):
        if not os.path.isabs(self.export_root):
            file_dir=os.path.dirname(os.path.realpath(__file__))
            out_path = os.path.join(file_dir,self.export_root)
        else:
            out_path=self.export_root

        # directory = os.path.join(out_path,self.simData.name)
        # try:
        #     os.makedirs(directory)
        # except OSError as e:
        #     if e.errno != errno.EEXIST:
        #         raise

        file = Path(out_path)/filename
        tree = ElementTree.ElementTree(self.cfg)
        # import pdb; pdb.set_trace()
        tree.write(file)



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