from Box2D import (b2World, b2Body, b2Contact)
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import os, errno
from ..types import SimData

def body_2_xml(b: b2Body):
    try:
        id = b.userData.id
        px = b.position.x
        py = b.position.y
        m = b.mass
        i = b.inertia
        vx = b.linearVelocity.x
        vy = b.linearVelocity.y
        #possibly wrong, according to https://www.packtpub.com/mapt/book/game_development/9781784394905/5/ch05lvl1sec51/angular-and-linear-damping
        #TODO: find definition of spin
        spin = b.angularVelocity
        orientation = b.angle
        shape = "circle"

        body = Element('body')
        body.set("index", str(id))
        if m > 0:
            body.set("type", "free")
        else:
            body.set("type", "fixed")
        pos = SubElement(body, "position")
        pos.set("x", str(px))
        pos.set("y", str(py))
        velocity = SubElement(body, "velocity")
        velocity.set("vx", str(vx))
        velocity.set("vy", str(vy))
        o = SubElement(body, "orientation")
        o.set("theta", str(orientation))
        mass = SubElement(body, "mass")
        mass.set("value", str(m))
        inertia = SubElement(body, "inertia")
        inertia.set("value", str(i))
        s = SubElement(body, "spin")
        s.set("omega", str(spin))
        s = SubElement(body, "shape")
        s.set("value", str(shape))
        return body
    except AttributeError:
        print("body without id encountered")
        raise


def contact_2_xml(c: b2Contact,c_ix=None):
    try:
        # TODO: what is this? divide the normal by its norm? normal as direction-vector only?
        t = 0
        n = 0
        depth = 0
        master = c.fixtureA.body.userData.id
        slave = c.fixtureB.body.userData.id
        px = c.manifold.localPoint.x
        py = c.manifold.localPoint.y
        nx = c.manifold.localNormal.x
        ny = c.manifold.localNormal.y
        contact = Element('contact')
        contact.set("index", str(c_ix))
        contact.set("master", str(master))
        contact.set("slave", str(slave))
        pos = SubElement(contact, "position")
        pos.set("x", str(px))
        pos.set("y", str(py))
        normal = SubElement(contact, "velocity")
        normal.set("nx", str(nx))
        normal.set("ny", str(ny))
        f = SubElement(contact, "force")
        f.set("n", str(n))
        f.set("t", str(t))
        d = SubElement(contact, "depth")
        d.set("value", str(depth))
        return contact
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
    def __init__(self,world:b2World, export_root, simData : SimData):
        self.world = world
        self.config_id = simData.name
        self.export_root = export_root
        self.simData=simData

    def snapshot(self):
        cfg = Element('configuration')
        cfg.set("name",self.config_id)
        cfg.set("time",str(0))
        for b in self.world.bodies:
            xb = body_2_xml(b)
            if xb is not None:
                cfg.append(xb)
        c_ix=-1
        for c in self.world.contacts:
            c_ix += 1
            xc = contact_2_xml(c,c_ix)
            if xc is not None:
                cfg.append(xc)
        return cfg

    def save_snapshot(self):
        xml = self.snapshot()
        xml = prettify(xml)
        directory = os.path.join(self.export_root,self.config_id)
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        file = os.path.join(directory,str(self.simData.t)+"s.xml")
        with open(file,"w") as f:
            f.write(xml)
