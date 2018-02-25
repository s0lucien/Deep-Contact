from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom


def body_2_xml(body):
    body_xml = Element('body')
    body_xml.set('index', str(body.userData))

    m = body.mass
    if m > 0:
        body_xml.set('type', 'free')
    else:
        body_xml.set('type', 'fixed')

    # mass
    mass = SubElement(body_xml, 'mass')
    mass.set('value', str(m))

    # position
    pos = SubElement(body_xml, 'position')
    pos.set('x', str(body.position.x))
    pos.set('y', str(body.position.y))

    # velocity
    v = SubElement(body_xml, 'velocity')
    v.set('x', str(body.linearVelocity.x))
    v.set('y', str(body.linearVelocity.y))

    # orientation
    ori = SubElement(body_xml, 'orientation')
    ori.set('theta', str(body.angle))

    # inertia
    # FIXME: p = m * v
    inertia = SubElement(body_xml, 'inertia')
    inertia.set('value', str(body.inertia))

    # spin
    # FIXME: Do we actually need spin? I do not think so
    spin = SubElement(body_xml, 'spin')
    spin.set('omega', str(body.angularVelocity))

    # shape
    # FIXME: Do we need shape? Since there are only moving circle

    return body_xml


def bodies_2_xml(bodies):
    bodies_xml = Element('bodies')
    bodies_xml.extend([
        body_2_xml(body)
        for body in bodies
    ])

    return bodies_xml
