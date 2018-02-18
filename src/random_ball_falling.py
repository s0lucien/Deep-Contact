#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        random_vector = lambda: b2Vec2(
            b2Random(xlow, xhi), b2Random(ylow, yhi)
        )

        circle = b2FixtureDef(
            shape=b2CircleShape(radius=1),
            density=1,
            friction=0.3,
        )

        self.using_contacts = True

        for i in range(100):
            self.world.CreateDynamicBody(
                fixtures=circle,
                position=random_vector(),
            )

    def Step(self, settings):
        super(FallingBall, self).Step(settings)


if __name__ == '__main__':
    main(FallingBall)
