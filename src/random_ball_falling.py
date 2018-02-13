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
)

import numpy as np


class FallingBall(Framework):

    bodies = []

    def __init__(self):
        super(FallingBall, self).__init__()

        ground  = self.world.CreateBody(
            shapes=b2LoopShape(
                vertices=[(20, 0), (20, 40), (-20, 40), (-20, 0)]
            )
        )

        circle = b2FixtureDef(
            shape=b2CircleShape(radius=0.5),
            density=1,
            friction=0.3,
        )

        for i in range(100):
            self.world.CreateDynamicBody(
                fixtures=circle,
                position=(
                    40*np.random.random_sample()-20,
                    40*np.random.random_sample(),
                ),
            )

    def Step(self, settings):
        super(FallingBall, self).Step(settings)


if __name__ == '__main__':
    main(FallingBall)
