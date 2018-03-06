from Box2D import (b2World, b2Body, b2Contact,b2_dynamicBody)
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import os, errno
from sim_types import SimData

def body_2_xml(body: b2Body):
    try:
        body_xml = Element('body')
        body_xml.set('index', str(body.userData.id))

        body_xml.set('type', 'free' if body.type is b2_dynamicBody else 'fixed')

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

        inertia = SubElement(body_xml, 'inertia')
        inertia.set('value', str(body.inertia))

        # spin
        spin = SubElement(body_xml, 'spin')
        spin.set('omega', str(body.angularVelocity))

        # shape
        shape = SubElement(body_xml, 'shape')
        shape.set('value', ' '.join(str(body.fixtures[0].shape).split()).replace('\n', ' ').replace('\r', ''))
        return body_xml
    except AttributeError:
        print("body without id encountered")
        raise


def contact_2_xml(contact: b2Contact,c_ix=None):
    try:
        contact_xmls = []
        for i in range(contact.manifold.pointCount):
            point = contact.worldManifold.points[i]
            normal = contact.worldManifold.normal
            manifold_point = contact.manifold.points[i]
            impulse = (manifold_point.normalImpulse, manifold_point.tangentImpulse)

            xml_contact = Element('contact')
            # master
            xml_contact.set('index', str(c_ix + i))
            xml_contact.set("master", str(contact.fixtureA.body.userData.id))
            xml_contact.set("slave", str(contact.fixtureB.body.userData.id))

            # position
            position = SubElement(xml_contact, 'position')
            position.set('x', str(point[0]))
            position.set('y', str(point[1]))

            xml_normal = SubElement(xml_contact, "normal")
            xml_normal.set("nx", str(normal.x))
            xml_normal.set("ny", str(normal.y))
            xml_impulse = SubElement(xml_contact, "impulse")
            xml_impulse.set("n", str(impulse[0]))
            xml_impulse.set("t", str(impulse[1]))
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

    def snapshot(self):
        cfg = Element('configuration')
        cfg.set("name",str(self.simData.name))
        cfg.set("time",str(self.simData.sim_t))
        for b in self.world.bodies:
            xb = body_2_xml(b)
            if xb is not None:
                cfg.append(xb)
        c_ix=0
        for c in self.world.contacts:
            xcs = contact_2_xml(c,c_ix)
            if xcs is not None:
                for xc in xcs:
                    cfg.append(xc)
            c_ix += c.manifold.pointCount
        return cfg

    def save_snapshot(self):
        xml = self.snapshot()
        xml = prettify(xml)
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
        file = os.path.join(directory,str(self.simData.step)+".xml")
        with open(file,"w") as f:
            f.write(xml)
