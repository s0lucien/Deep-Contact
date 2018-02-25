#!/usr/bin/env python
# -*- coding: utf-8 -*-
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom


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
    # FIXME: Do we need shape? Since there are only moving circle
    shape = SubElement(body_xml, 'shape')
    if m > 0:
        shape.set('value', 'circle')
    else:
        shape.set('value', 'ground')

    return body_xml


def _contact_2_xml(contact, index):
    contact_xml = Element('contact')
    contact_xml.set('index', str(index))
    contact.set("master", str(contact.fixtureA.body.userData))
    contact.set("slave", str(contact.fixtureB.body.userData))

    # position
    pos = SubElement(contact_xml, "position")
    pos.set('x', str(contact.manifold.localPoint.x))
    pos.set("y", str(contact.manifold.localPoint.y))

    # velocity
    velocity = SubElement(contact_xml, 'velocity')
    velocity.set('nx', str(contact.manifold.localPoint.x))
    velocity.set('ny', str(contact.manifold.localPoint.y))

    # force
    force = SubElement(contact_xml, 'force')
    force.set('n', str(n))
    force.set('t', str(t))

    # depth
    d = SubElement(contact_xml, 'depth')
    d.set('value', str(depth))

    return contact_xml


def config_xml(bodies, contacts):
    config_xml = Element('Configuration')
    config_xml.extend([
        _body_2_xml(body)
        for body in bodies
    ])

    config_xml.extend([
        _contact_2_xml(contact)
        for contact in contacts
    ])

    return config_xml
