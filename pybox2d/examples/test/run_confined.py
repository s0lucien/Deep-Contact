from own.framework import (Framework, Keys, main)
from Box2D import (b2CircleShape, b2LoopShape, b2FixtureDef)
from own.types import BodyData
from own.gen_world import create_circle


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

        # The bodies
        radius = 0.5
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


if __name__ == "__main__":
    main(Confined)
