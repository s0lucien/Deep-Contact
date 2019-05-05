from own.framework import (Framework, Keys, main)
from Box2D import (b2CircleShape, b2LoopShape, b2FixtureDef,b2Vec2)
from own.types import BodyData
from own.gen_world import GenRandomCirclesWorld


class Confined(Framework):

    def __init__(self):
        super(Confined, self).__init__()
        self.name = "Random balls falling"
        xlow, xhi = -20, 20
        ylow, yhi = 0, 40

        # The ground
        ground = self.world.CreateBody(
            shapes=b2LoopShape(
                vertices=[(xhi, ylow), (xhi, yhi), (xlow, yhi), (xlow, ylow)]
            )
        )

        gen = GenRandomCirclesWorld(self.world)
        gen.new(100, b2Vec2(xlow,ylow),  b2Vec2(xhi,yhi), 0.5, 2)
        self.world.gravity = (0, -9.81)

        b_ix = -1
        for b in self.world.bodies:
            b_ix += 1
            b.userData = BodyData(b_ix)


if __name__ == "__main__":
    main(Confined)
