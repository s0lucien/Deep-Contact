

from ..framework import (Framework, main)
from ..gen_world import new_confined_clustered_circles_world
from Box2D import (b2LoopShape, b2Vec2)


class Confined(Framework):

    def __init__(self):
        super(Confined, self).__init__()
        self.name = "Random balls centre falling"
        xlow, xhi = -20, 20
        ylow, yhi = 0, 40
        n_circles = 100
        sigma_coef = 1.3

        new_confined_clustered_circles_world(self.world, n_circles,
                                             p_ll=b2Vec2(xlow,ylow),
                                             p_hr=b2Vec2(xhi,yhi),
                                             radius_range=(1,1), sigma=sigma_coef,
                                             seed=None)
        print("finished world generation -- break here if you need to :)")

    def Step(self, settings):
        super(Confined, self).Step(settings)


if __name__ == "__main__":
    main(Confined)
