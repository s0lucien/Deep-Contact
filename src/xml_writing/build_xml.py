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


def _contact_2_xml(contact_point, index):
    contact_xml = Element('contact')
    contact_xml.set('index', str(index))
    contact_xml.set("master", str(contact_point['fixtureA']))
    contact_xml.set("slave", str(contact_point['fixtureB']))

    # position
    pos = SubElement(contact_xml, "position")
    pos.set('x', str(contact_point['position'][0]))
    pos.set("y", str(contact_point['position'][1]))

    # normal
    normal = SubElement(contact_xml, 'normal')
    normal.set('nx', str(contact_point['normal'][0]))
    normal.set('ny', str(contact_point['normal'][1]))

    # force
    force = SubElement(contact_xml, 'force')
    # normal Impulse
    # force.set('n', str(contact.noramlImpulse))
    # tangent Impulse
    force.set('t', str(0))

    # depth
    d = SubElement(contact_xml, 'depth')
    d.set('value', str(0))

    return contact_xml


def config_xml(bodies, contact_points, stepCount, timeStep):
    config_xml = Element('Configuration')
    config_xml.set('name', str(stepCount))
    config_xml.set('time', str(timeStep))
    config_xml.extend([
        _body_2_xml(body)
        for body in bodies
    ])

    config_xml.extend([
        _contact_2_xml(contact_point, i)
        for i, contact_point in enumerate(contact_points)
    ])

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
