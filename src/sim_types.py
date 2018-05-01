from time import time
from Box2D import b2FixtureDef, b2CircleShape, b2LoopShape, b2World


class BodyData:
    def __init__(self, b_id=None, shape=None):
        self.id = b_id
        self.shape = shape


class SimData:
    def __init__(self, name="dcSim2D", d_t=0):
        self.name = name
        self.dt = d_t

        self.step = 0
        self.sim_t = 0
        self.wall_t = 0

        self.init_t=time()
        self.ticking=False


    def tick(self, s=1):
        self.ticking=True
        self.step += s
        self.sim_t += s * self.dt
        self.init_t = time()


    def tock(self):
        if self.ticking:
            self.wall_t += time() - self.init_t
            self.init_t = time()
            self.ticking=False


class GenWorld:
    '''
    Container class for our world generator . Sets default parameters of the simulation
    which should be constants from our experiment point of view
    '''
    def __init__(self, world:b2World):
        self.world = world
        self.world.gravity = (0, -9.81)
        self.world.enableWarmStarting = True
        self.world.enableContinous = False
        self.world.allowSleeping = False
        try:
            if self.world.initialized is True:
                raise Exception("The world has already been populated.")
        except AttributeError:
            self.world.initialized = False

    def fill(self):
        '''
        will generate bodies inside the world
        :return:
        '''
        if self.world.initialized:
            raise Exception("The world is locked. running more generators is not possible.")
        pass


class dcCircleShape:
    '''
    a wrapper class used for storing/loading circles to xml and back
    '''
    def __init__(self, radius):
        self.radius = radius
        self.fixture = b2FixtureDef(shape=b2CircleShape(radius=self.radius,
                                                   pos=(0, 0)),
                               density=1, friction=0.5, restitution=0)

    def __repr__(self):
        return "dcCircleShape(radius={0})".format(self.radius)


class dcLoopShape:
    '''
    a wrapper class used for storing/loading boxes to xml and back
    '''
    def __init__(self,vertices):
        self.vertices = vertices
        assert isinstance(vertices, list)
        self.fixture = b2FixtureDef(shape=b2LoopShape(vertices=self.vertices),
                                    density=1, friction=0.5, restitution=0)

    def __repr__(self):
        return "dcLoopShape(vertices={0})".format(self.vertices)
