from Box2D import b2World, b2FixtureDef, b2CircleShape, b2Vec2
import numpy as np
import logging

def create_circle(world, pos, radius):
    fixture = b2FixtureDef(shape=b2CircleShape(radius=radius,
                                               pos=(0, 0)),
                           density=1, friction=0.1)

    world.CreateDynamicBody(
        position=pos,
        fixtures=fixture,
    )


# p_ll - position lower left , p_hr - position higher right
class GenRandomCirclesWorld:
    def __init__(self, world: b2World):
        self.world = world
        self.max_tries = 10

    def new(self, n, p_ll:b2Vec2, p_hr:b2Vec2, min_radius, max_radius):
        circ = np.empty((0,3))
        len_x, len_y = p_hr - p_ll
        sz = min_radius + np.random.rand(n) * (max_radius - min_radius)
        i = 0
        failed = 0
        while i < n:
            p = p_ll + np.random.rand(2) * (len_x,len_y)
            rad = sz[i]
            #goes out of the box - not too good
            if np.linalg.norm(p-p_ll) < rad or np.linalg.norm(p_hr-p) < rad:
                failed+=1
                continue
            #TODO: overlaps other circle -  not good
            overlap=False
            if overlap is True:
                failed +=1
                continue
            if failed > self.max_tries:
                failed=True
                break
            c = np.array([p.x, p.y, rad])
            circ = np.vstack((circ,c))
            i += 1
        # done with the circle generation, now populate the world
        for i in range(circ.shape[0]):
            c = circ[i,:]
            logging.info("creating circle ", c)
            create_circle(self.world,c[0:2],c[2])
        if failed is True:
            raise Exception("Unable to place a circle after " + str(self.max_tries) + ". Only " + str(i)
                            + " out of " + str(n) + " circles could fit")

#
# class GenCenteredAroundCenterCirclesWorld:
#     def __init__(self, world: b2World, sigma=4):
#         self.world = world
#
#     def new(self, n, p_ll, p_hr, min_radius, max_radius):
#         i = -1
#         circ = []
#         while i < n:
#             i += 1
