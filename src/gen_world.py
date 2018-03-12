from Box2D import b2World, b2FixtureDef, b2CircleShape, b2Vec2
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

def create_circle(world, pos, radius):
    fixture = b2FixtureDef(shape=b2CircleShape(radius=radius,
                                               pos=(0, 0)),
                           density=1, friction=0.5, restitution=0)

    world.CreateDynamicBody(
        position=pos,
        fixtures=fixture,
    )


# p_ll - position lower left , p_hr - position higher right
class GenRandomCirclesWorld:
    def __init__(self, world: b2World):
        self.world = world
        # this is simply brute-force. For reasonably high n, we run into
        #https://en.wikipedia.org/wiki/Coupon_collector%27s_problem
        self.max_consec_tries = 50

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


class GenClusteredCirclesWorld:
    def __init__(self, world: b2World, sigma=None, mu=None, seed=None):
        np.random.seed(seed) # Might misbehave if running multiple world generators simultaneously
        self.world = world
        self.sigma = sigma
        self.max_consec_tries = 100
        self.mu=mu


    def new(self, n, p_ll:b2Vec2, p_hr:b2Vec2, min_radius, max_radius,sigma_coef=1):
        if self.mu==None:
            mu = (p_hr + p_ll)/2
        else :
            mu = self.mu
        if self.sigma == None:
            sigma = np.sqrt((p_hr - p_ll)/2)*sigma_coef
        else:
            sigma = self.sigma
        circ = np.empty((0,3))
        sz = min_radius + np.random.rand(n) * (max_radius - min_radius)
        i = 0
        failed = 0
        while i < n:
            p = b2Vec2(np.random.normal(mu, sigma, 2))
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
