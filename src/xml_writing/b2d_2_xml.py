from Box2D import (b2World, b2Body, b2Contact)
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import os, errno
from sim_types import SimData

def body_2_xml(b: b2Body):
    try:
        id = b.userData.id
        px = b.position.x
        py = b.position.y
        m = b.mass
        i = b.inertia
        vx = b.linearVelocity.x
        vy = b.linearVelocity.y
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
        p1 = p2 = c.manifold.points[0].localPoint
        t1 = t2 = c.manifold.points[0].tangentImpulse
        n1 = n2 = c.manifold.points[0].normalImpulse
        if c.manifold.pointCount>1 : # B2D returns either 1 or 2 points
            p2 = c.manifold.points[1].localPoint
            t2 = c.manifold.points[1].tangentImpulse
            n2 = c.manifold.points[1].normalImpulse
        depth = None
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
        normal = SubElement(contact, "normal")
        normal.set("nx", str(nx))
        normal.set("ny", str(ny))
        f1 = SubElement(contact, "force")
        f1.set("n", str(n1))
        f1.set("t", str(t1))
        f1.set("px", str(p1.x))
        f1.set("py", str(p1.y))
        f2 = SubElement(contact, "force")
        f2.set("n", str(n2))
        f2.set("t", str(t2))
        f2.set("px", str(p2.x))
        f2.set("py", str(p2.y))
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
        if not os.path.isabs(self.export_root):
            file_dir=os.path.dirname(os.path.realpath(__file__))
            out_path = os.path.join(file_dir,self.export_root)
        else:
            out_path=self.export_root
        directory = os.path.join(out_path,self.config_id)
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        file = os.path.join(directory,str(self.simData.t)+"s.xml")
        with open(file,"w") as f:
            f.write(xml)
