

from ..framework import (Framework, main)
from ..sim_types import BodyData
from ..gen_world import GenClusteredCirclesWorld
from Box2D import (b2LoopShape, b2Vec2)


class Confined(Framework):

    def __init__(self):
        super(Confined, self).__init__()
        self.name = "Random balls centre falling"
        xlow, xhi = -20, 20
        ylow, yhi = 0, 40

        # The ground
        ground = self.world.CreateBody(
            shapes=b2LoopShape(
                vertices=[(xhi, ylow), (xhi, yhi), (xlow, yhi), (xlow, ylow)]
            )
        )

        gen = GenClusteredCirclesWorld(self.world)
        sigma_coef = 1.2
        gen.new(100, b2Vec2(xlow,ylow),  b2Vec2(xhi,yhi), 1, 1, sigma_coef)
        self.world.gravity = (0, -9.81)

        b_ix = 0
        for b in self.world.bodies:
            b.userData = BodyData(b_ix)
            b_ix += 1


    def Step(self, settings):
        super(Confined, self).Step(settings)


if __name__ == "__main__":
    main(Confined)
