#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

from .framework import (
    Framework,
    Keys,
    main,
)

from Box2D import (
    b2CircleShape,
    b2LoopShape,
    b2_pi,
    b2FixtureDef,
    b2Random,
    b2Vec2,
)

import numpy as np


class FallingBall(Framework):

    bodies = []

    def __init__(self):
        super(FallingBall, self).__init__()

        xlow, xhi = -20, 20
        ylow, yhi = 0, 40

        ground  = self.world.CreateBody(
            shapes=b2LoopShape(
                vertices=[(xhi, ylow), (xhi, yhi), (xlow, yhi), (xlow, ylow)]
            )
        )
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
            position = random_vector()
            good = True
            for prepared_position in positions:
                if distance(position, prepared_position) < 2:
                    good = False
                    break
            if good:
                self.world.CreateDynamicBody(
                    fixtures=circle,
                    position=position,
                )
                positions.append(position)

    def Step(self, settings):
        super(FallingBall, self).Step(settings)


def distance(point_1, point_2):
    return np.sqrt(
        (point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2
    )


if __name__ == '__main__':
    main(FallingBall)
