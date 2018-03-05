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
    b2LoopShape,
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

        ground = self.world.CreateStaticBody(
            shapes=[
                b2LoopShape(
                    vertices=[
                        (xlow, ylow), (xhi, ylow), (xhi, yhi), (xlow, yhi)])
            ],
            userData=0,
        )

        index = 0

        self.bodies.append(ground)
        random_vector = lambda: (
            b2Random(xlow+1, xhi-1), b2Random(ylow+1, yhi-1)
        )

        radius = 1.2
        circle = b2FixtureDef(
            shape=b2CircleShape(radius=radius),
            density=1,
            friction=0.5,
        )

        positions = []
        while len(positions)<100:
            index += 1
            position = normal_random()

            if position[0] < xlow or position[0] > xhi:
                break
            if position[1] < ylow or position[1] > yhi:
                break

            good = True
            for prepared_position in positions:
                if distance(position, prepared_position) < 2 * radius:
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
        here = os.path.dirname('__file__')
        dir_name = os.path.dirname(here)

        super(FallingBall, self).Step(settings)
        timeStep = 1 / settings.hz * self.stepCount
        if settings.config_build:
            config = Configuration(
                bodies=self.bodies,
                contact_points=self.world.contacts,
                stepCount=self.stepCount,
                timeStep=timeStep,
            )
            config.build_xml(settings.export_path)


def distance(point_1, point_2):
    return np.sqrt(
        (point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2
    )


def normal_random():
    mean = [0, 20]
    cov = [[20, 0], [0, 20]]

    return np.random.multivariate_normal(mean, cov)


if __name__ == '__main__':
    main(FallingBall)
