from Box2D import b2World, b2FixtureDef, b2CircleShape, b2Vec2
import numpy as np
import logging
from sim_types import dcCircleShape, GenWorld, dcLoopShape , BodyData

logging.basicConfig(level=logging.INFO)


def create_circle(world, pos, radius):
    shape = dcCircleShape(radius)

    circ = world.CreateDynamicBody(
        position=pos,
        fixtures=shape.fixture,
    )
    circ.userData = BodyData()
    circ.userData.shape = str(shape)


# p_ll - position lower left , p_hr - position higher right
def create_fixed_box(world,p_ll:b2Vec2, p_hr:b2Vec2,pos=(0,0)):
    xlow, ylow = p_ll
    xhi, yhi = p_hr
    shape = dcLoopShape([(xhi, ylow), (xhi, yhi), (xlow, yhi), (xlow, ylow)])

    box = world.CreateStaticBody(
        position=pos,
        fixtures=shape.fixture
    )
    box.userData=BodyData()
    box.userData.shape = str(shape)


def new_confined_clustered_circles_world(world, n_bodies, p_ll , p_hr, radius_range, sigma, seed=None):
    '''
    Use this as the entry point generator. This uses the others classes in this file to generate a world
    radius_range = 2d tuple/array that holds minimum and maximum range for the circles
    '''
    create_fixed_box(world,p_ll,p_hr)
    GenClusteredCirclesRegion(world, seed=seed).fill(n_bodies, p_ll, p_hr, radius_range, sigma)

    for i in range(world.bodyCount):
        world.bodies[i].userData.id = i
    world.initialized = True


class GenRandomCirclesRegion(GenWorld):
    def __init__(self, world: b2World, seed=None):
        super(GenRandomCirclesRegion, self).__init__(world)
        self.seed = seed
        self.random = np.random.RandomState(self.seed)
        # this is simply brute-force. For reasonably high n, we run into
        #https://en.wikipedia.org/wiki/Coupon_collector%27s_problem
        self.max_consec_tries = 50

    def fill(self, n, p_ll:b2Vec2, p_hr:b2Vec2, radius_range):
        super(GenRandomCirclesRegion, self).fill()
        min_radius, max_radius = radius_range
        circ = np.empty((0,3))
        len_x, len_y = p_hr - p_ll
        sz = min_radius + self.random.rand(n) * (max_radius - min_radius)
        i = 0
        failed = 0
        while i < n:
            p = p_ll + self.random.rand(2) * (len_x,len_y)
            rad = sz[i]
            #goes out of the box - not too good
            if np.any(np.asarray([p_hr - p, p-p_ll]).flatten() < rad):
                logging.info("Out of the box")
                failed+=1
                if failed > self.max_consec_tries:
                    failed = True
                    break
                continue
            #overlaps other circle -  not good
            P=circ[:,0:2]
            dist = np.linalg.norm(P - np.asarray(p), axis=1)
            R=circ[:,2]
            no_overlap=np.all(R+rad<dist)
            if no_overlap ==  False:
                logging.info("Overlap detected")
                failed +=1
                if failed > self.max_consec_tries:
                    failed = True
                    break
                continue
            c = np.array([p.x, p.y, rad])
            circ = np.vstack((circ,c))
            failed = 0
            i += 1
        # done with the circle generation, now populate the world
        for i in range(circ.shape[0]):
            c = circ[i,:]
            logging.debug("creating circle " + str(c))
            create_circle(self.world,c[0:2],c[2])
        if failed is True:
            raise Exception("Unable to place a circle after " + str(self.max_consec_tries) + "tries . Only " + str(i)
                            + " out of " + str(n) + " circles could fit")


class GenClusteredCirclesRegion(GenWorld):
    def __init__(self, world: b2World, sigma=None, mu=None, seed=None):
        super(GenClusteredCirclesRegion, self).__init__(world)
        self.seed = seed
        self.random = np.random.RandomState(self.seed)
        self.sigma = sigma
        self.max_consec_tries = 100
        self.mu=mu

    def fill(self, n, p_ll:b2Vec2, p_hr:b2Vec2, radius_range ,sigma_coef=1):
        super(GenClusteredCirclesRegion, self).fill()
        min_radius, max_radius = radius_range
        if self.mu==None:
            mu = (p_hr + p_ll)/2
        else :
            mu = self.mu
        if self.sigma == None:
            sigma = np.sqrt((p_hr - p_ll)/2)*sigma_coef
        else:
            sigma = self.sigma
        circ = np.empty((0,3))
        sz = min_radius + self.random.rand(n) * (max_radius - min_radius)
        i = 0
        failed = 0
        while i < n:
            p = b2Vec2(self.random.normal(mu, sigma, 2))
            rad = sz[i]
            #goes out of the box - not too good
            if np.any(np.asarray([p_hr - p, p - p_ll]).flatten() < rad):
                logging.info("Out of the box")
                failed+=1
                if failed > self.max_consec_tries:
                    failed = True
                    break
                continue
            #overlaps other circle -  not good
            P=circ[:,0:2]
            dist = np.linalg.norm(P - np.asarray(p), axis=1)
            R=circ[:,2]
            no_overlap=np.all(R+rad<dist)
            if no_overlap ==  False:
                logging.info("Overlap detected")
                failed +=1
                if failed > self.max_consec_tries:
                    failed = True
                    break
                continue
            c = np.array([p.x, p.y, rad])
            circ = np.vstack((circ,c))
            failed = 0
            i += 1
        # done with the circle generation, now populate the world
        for i in range(circ.shape[0]):
            c = circ[i,:]
            logging.debug("creating circle " + str(c))
            create_circle(self.world,c[0:2],c[2])
        if failed is True:
            raise Exception("Unable to place a circle after " + str(self.max_consec_tries) + "tries . Only " + str(i)
                            + " out of " + str(n) + " circles could fit")
