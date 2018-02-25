#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

from .framework import (
    Framework,
    Keys,
    main,
)
from .xml_writing.build_xml import Configuration

from Box2D import (
    b2CircleShape,
    b2LoopShape,
    b2_pi,
    b2FixtureDef,
    b2Random,
    b2Vec2,
)

import numpy as np
import os


class FallingBall(Framework):

    bodies = []

    def __init__(self):
        super(FallingBall, self).__init__()
        self.using_contacts = True
        self.contacts = []

        xlow, xhi = -20, 20
        ylow, yhi = 0, 40

        index = 0

        ground  = self.world.CreateBody(
            shapes=b2LoopShape(
                vertices=[(xlow, yhi), (xlow, ylow), (xhi, ylow), (xhi, yhi)]
            ),
            userData = index,
        )
        self.bodies.append(ground)
        random_vector = lambda: (
            b2Random(xlow+1, xhi-1), b2Random(ylow+1, yhi-1)
        )

        circle = b2FixtureDef(
            shape=b2CircleShape(radius=1),
            density=1,
            friction=0.3,
        )

        self.using_contacts = True

        positions = []
        while len(positions)<100:
            index += 1
            position = random_vector()
            good = True
            for prepared_position in positions:
                if distance(position, prepared_position) < 2:
                    good = False
                    break
            if good:
                self.bodies.append(
                    self.world.CreateDynamicBody(
                        fixtures=circle,
                        position=position,
                        userData=index,
                    )
                )
                positions.append(position)

    def PreSolve(self, contact, old_manifold):
        super(FallingBall, self).PreSolve(contact, old_manifold)
        self.contacts.append(contact)


    def Step(self, settings):
        super(FallingBall, self).Step(settings)
        timeStep = 1 / settings.hz * self.stepCount
        config = Configuration(
            bodies=self.bodies,
            contacts=self.contacts,
            stepCount=self.stepCount,
            timeStep=timeStep,
        )
        print(config.build_xml(
            export_path='~/KU/Deep-Contact/xml'))
        contacts=[]


def distance(point_1, point_2):
    return np.sqrt(
        (point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2
    )


if __name__ == '__main__':
    main(FallingBall)
