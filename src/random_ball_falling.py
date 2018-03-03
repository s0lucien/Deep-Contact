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
    b2EdgeShape,
    b2CircleShape,
    b2_pi,
    b2FixtureDef,
    b2Random,
    b2Vec2,
    b2GetPointStates,
)

import numpy as np
import os


class FallingBall(Framework):

    bodies = []
    name = 'Exp for 2D in Deep Contact'

    def __init__(self):
        super(FallingBall, self).__init__()

        self.using_contacts = True
        self.contacts = []

        xlow, xhi = -20, 20
        ylow, yhi = 0, 40

        ground_1 = self.world.CreateStaticBody(
            shapes=[b2EdgeShape(vertices=[(xlow, ylow), (xhi, ylow)])],
            userData=0,
        )
        ground_2 = self.world.CreateStaticBody(
            shapes=[b2EdgeShape(vertices=[(xlow, ylow), (xlow, yhi)])],
            userData=1,
        )
        ground_3 = self.world.CreateStaticBody(
            shapes=[b2EdgeShape(vertices=[(xhi, ylow), (xhi, yhi)])],
            userData=2,
        )

        index = 2
        grounds= [ground_1, ground_2, ground_3]

        self.bodies.extend(grounds)
        random_vector = lambda: (
            b2Random(xlow+1, xhi-1), b2Random(ylow+1, yhi-1)
        )

        circle = b2FixtureDef(
            shape=b2CircleShape(radius=1),
            density=1,
            friction=0.3,
        )

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

    def Step(self, settings):
        super(FallingBall, self).Step(settings)
        timeStep = 1 / settings.hz * self.stepCount
        if settings.config_build:
            config = Configuration(
                bodies=self.bodies,
                contact_points=self.world.contacts,
                stepCount=self.stepCount,
                timeStep=timeStep,
            )
            config.build_xml(export_path='/home/jwu/KU/Deep-Contact/xml')
        contacts=[]
        print(self.world.contacts)


def distance(point_1, point_2):
    return np.sqrt(
        (point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2
    )


if __name__ == '__main__':
    main(FallingBall)
