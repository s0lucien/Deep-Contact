#!/usr/bin/env python
# -*- coding: utf-8 -*-
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import os


def _body_2_xml(body):
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
    shape = SubElement(body_xml, 'shape')
    if m > 0:
        shape.set('value', 'circle')
    else:
        shape.set('value', 'ground')

    return body_xml


def _contact_2_xml(contact, index):
    contact_xmls = []
    for i in range(contact.manifold.pointCount):
        point = contact.worldManifold.points[i]
        normal = contact.worldManifold.normal
        manifold_point = contact.manifold.points[i]
        impulse = (manifold_point.normalImpulse, manifold_point.tangentImpulse)

        contact_xml = Element('contact')
        # master
        contact_xml.set('index', index + i)
        contact_xml.set("master", contact.fixtureB.body)
        contact_xml.set("slave", contact.fixtureB.body)
        contact_xml.set('master_shape', contact.fixtureA.shape)
        contact_xml.set('slave_shape', contact.fixtureB.shape)

        # position
        position = SubElement(contact_xml, 'position')
        position.set('x', str(point[0]))
        position.set('y', str(point[1]))

        # normal
        xml_normal = SubElement(contact_xml, 'normal')
        xml_normal.set('normal', normal)

        # Impulse
        xml_impulse = SubElement(contact_xml, 'impulse')
        xml_impulse.set('n', impulse[0])
        xml_impulse.set('t', impulse[1])

        contact_xmls.append(contact_xml)

    return contact_xmls, contact.manifold.pointCount


def config_xml(bodies, contacts, stepCount, timeStep):
    config_xml = Element('Configuration')
    config_xml.set('name', str(stepCount))
    config_xml.set('time', str(timeStep))
    config_xml.extend([
        _body_2_xml(body)
        for body in bodies
    ])

    num_contact_point = 0
    for i, contact in enumerate(contacts):
        contact_xmls, num = _contact_2_xml(contact, i)
        config_xml.extend(contact_xmls)
        num_contact_point += num

    return config_xml


class Configuration():
    def __init__(self,
                 bodies,
                 contact_points,
                 stepCount,
                 timeStep):
        self.bodies = bodies
        self.contact_points = contact_points
        self.stepCount = stepCount
        self.timeStep = timeStep

    def build_xml(self, export_path):
        configuration = config_xml(self.bodies,
                                self.contact_points,
                                self.stepCount,
                                self.timeStep)
        xml = prettify(configuration)
        file_path = os.path.join(
            export_path,
            '{}.xml'.format(self.stepCount))

        with open(file_path, 'w') as f:
            f.write(xml)
        print('Training data {}.xml has been created'.format(self.stepCount))


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
