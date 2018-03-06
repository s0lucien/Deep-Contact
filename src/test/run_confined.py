from ..framework import (
    Framework,
    Keys,
    main,
)
from Box2D import (b2LoopShape)
from sim_types import BodyData
from gen_world import create_circle


class Confined(Framework):

    def __init__(self):
        super(Confined, self).__init__()
        self.name = "Stacked balls falling"
        xlow, xhi = -20, 20
        ylow, yhi = 0, 40

        # The ground
        ground = self.world.CreateBody(
            shapes=b2LoopShape(
                vertices=[(xhi, ylow), (xhi, yhi), (xlow, yhi), (xlow, ylow)]
            )
        )

        # The bodies
        radius = 1
        columnCount = 5
        rowCount = 5

        for j in range(columnCount):
            for i in range(rowCount):
                create_circle(self.world,
                              (-10 + (2.1 * j + 1 + 0.01 * i) * radius, (2 * i + 1) * radius),
                              radius
                              )

        self.world.gravity = (0, -9.81)

        b_ix = -1
        for b in self.world.bodies:
            b_ix += 1
            b.userData = BodyData(b_ix)

    def Step(self, settings):
        super(Confined, self).Step(settings)

if __name__ == "__main__":
    main(Confined)
