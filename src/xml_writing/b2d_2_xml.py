from Box2D import (b2World, b2Body, b2Contact,b2_dynamicBody)
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import os, errno
from ..sim_types import SimData


def body_2_xml(body: b2Body):
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

    # position
    pos = SubElement(body_xml, 'position')
    pos.set('x', str(body.position.x))
    pos.set('y', str(body.position.y))

    # velocity
    v = SubElement(body_xml, 'velocity')
    v.set('vx', str(body.linearVelocity.x))
    v.set('vy', str(body.linearVelocity.y))

    # orientation
    ori = SubElement(body_xml, 'orientation')
    ori.set('theta', str(body.angle))

    # spin
    spin = SubElement(body_xml, 'spin')
    spin.set('omega', str(body.angularVelocity))

    # shape
    shape = SubElement(body_xml, 'shape')
    shape.set('value', str(body.userData.shape))

    return body_xml


def contact_2_xml(contact: b2Contact, c_ix=None):
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

        contact_xmls.append(ct_xml)

    return contact_xmls


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


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

        self.cfg.set("name",str(self.simData.name))
        self.cfg.set("time",str(self.simData.sim_t))
        self.cfg.set("dt",str(self.simData.dt))


    # Store all bodies in the world as xml
    def snapshot_bodies(self):
        for b in self.world.bodies:
            xb = body_2_xml(b)
            if xb is not None:
                self.bodies.append(xb)

    # Store the given contact as xml
    def snapshot_contact(self, contact:b2Contact):
        ct_xml = contact_2_xml(contact)
        for ct_pt in ct_xml:
            self.contacts.append(ct_pt)

    # Add the given impulse to the already-existing xml representing the given contact
    def snapshot_impulse(self, contact:b2Contact, impulse):
        for i in range(contact.manifold.pointCount):
            # We find the corresponding contact xml
            index = str(int(contact.userData) + i)
            ct_xml = self.contacts.findall(".//contact[@index='" + index + "']")[0]

            # We add the impulse
            impulse_xml = SubElement(ct_xml, "impulse")
            impulse_xml.set("n", str(impulse.normalImpulses[i]))
            impulse_xml.set("t", str(impulse.tangentImpulses[i]))


    # Save all the stored xml to a file
    def save_snapshot(self):
        xml = prettify(self.cfg)

        if not os.path.isabs(self.export_root):
            file_dir=os.path.dirname(os.path.realpath(__file__))
            out_path = os.path.join(file_dir,self.export_root)
        else:
            out_path=self.export_root

        directory = os.path.join(out_path,self.simData.name)
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        file = os.path.join(directory,self.simData.name+"_"+str(self.simData.step)+".xml")
        with open(file,"w") as f:
            f.write(xml)
