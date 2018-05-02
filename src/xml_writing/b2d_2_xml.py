from Box2D import (b2World, b2Body, b2Contact,b2_dynamicBody)
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import os, errno
from ..sim_types import SimData


def body_2_xml(body: b2Body):
    try:
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
    except AttributeError:
        print("body without id encountered")
        raise


def contact_2_xml(contact: b2Contact, c_ix=None):
    try:
        contact_xmls = []
        for i in range(contact.manifold.pointCount):
            # index
            xml_contact = Element('contact')
            xml_contact.set('index', str(c_ix + i))
            xml_contact.set("master", str(contact.fixtureA.body.userData.id))
            xml_contact.set("slave", str(contact.fixtureB.body.userData.id))

            # position
            point = contact.worldManifold.points[i]
            position = SubElement(xml_contact, 'position')
            position.set('x', str(point[0]))
            position.set('y', str(point[1]))

            # normal
            normal = contact.worldManifold.normal
            xml_normal = SubElement(xml_contact, "normal")
            xml_normal.set("nx", str(normal.x))
            xml_normal.set("ny", str(normal.y))

            contact_xmls.append(xml_contact)

        return contact_xmls
    except AttributeError:
        print("contact between body without id encountered")
        raise


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


class XMLExporter:
    def __init__(self,world:b2World, export_root):
        self.world = world
        self.export_root = export_root
        self.simData=world.userData
        self.__reset__()

    def __reset__(self):
        self.cfg = Element("configuration")
        self.cfg.append(Element("bodies"))
        self.cfg.append(Element("contacts"))

        self.cfg.set("name",str(self.simData.name))
        self.cfg.set("time",str(self.simData.sim_t))
        self.cfg.set("dt",str(self.simData.dt))


    def snapshot_bodies(self):
        bodies = self.cfg.find("bodies")
        for b in self.world.bodies:
            xb = body_2_xml(b)
            if xb is not None:
                bodies.append(xb)

    def snapshot_contacts(self):
        contacts = self.cfg.find("contacts")
        c_ix=0
        for c in self.world.contacts:
            xcs = contact_2_xml(c,c_ix)
            if xcs is not None:
                for xc in xcs:
                    contacts.append(xc)
            c_ix += c.manifold.pointCount


    def snapshot_impulses(self, impulses):
        for contact in self.cfg.find("contacts"):
            master = int(contact.get("master"))
            slave  = int(contact.get("slave"))

            imp = impulses.get((master, slave))
            if imp:
                impulse = SubElement(contact, "impulse")
                impulse.set("n", str(imp[0]))
                impulse.set("t", str(imp[1]))


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
