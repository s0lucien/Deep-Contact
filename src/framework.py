
"""
The framework's base is FrameworkBase. See its help for more information.
"""
from time import time

from Box2D import (b2World, b2AABB, b2CircleShape, b2Color, b2Vec2)
from Box2D import (b2ContactListener, b2DestructionListener, b2DrawExtended)
from Box2D import (b2Fixture, b2FixtureDef, b2Joint)
from Box2D import (b2GetPointStates, b2QueryCallback, b2Random)
from Box2D import (b2_addState, b2_dynamicBody, b2_epsilon, b2_persistState)

from .settings import fwSettings


class fwDestructionListener(b2DestructionListener):
    """
    The destruction listener callback:
    "SayGoodbye" is called when a joint or shape is deleted.
    """

    def __init__(self, test, **kwargs):
        super(fwDestructionListener, self).__init__(**kwargs)
        self.test = test

    def SayGoodbye(self, obj):
        if isinstance(obj, b2Joint):
            if self.test.mouseJoint == obj:
                self.test.mouseJoint = None
            else:
                self.test.JointDestroyed(obj)
        elif isinstance(obj, b2Fixture):
            self.test.FixtureDestroyed(obj)


class fwQueryCallback(b2QueryCallback):

    def __init__(self, p):
        super(fwQueryCallback, self).__init__()
        self.point = p
        self.fixture = None

    def ReportFixture(self, fixture):
        body = fixture.body
        if body.type == b2_dynamicBody:
            inside = fixture.TestPoint(self.point)
            if inside:
                self.fixture = fixture
                # We found the object, so stop the query
                return False
        # Continue the query
        return True


class Keys(object):
    pass


class FrameworkBase(b2ContactListener):
    """
    The base of the main testbed framework.

    If you are planning on using the testbed framework and:
    * Want to implement your own renderer (other than Pygame, etc.):
      You should derive your class from this one to implement your own tests.
      See empty.py or any of the other tests for more information.
    * Do NOT want to implement your own renderer:
      You should derive your class from Framework. The renderer chosen in
      fwSettings (see settings.py) or on the command line will automatically
      be used for your test.
    """
    name = "None"
    description = None
    TEXTLINE_START = 30
    colors = {
        'mouse_point': b2Color(0, 1, 0),
        'bomb_center': b2Color(0, 0, 1.0),
        'bomb_line': b2Color(0, 1.0, 1.0),
        'joint_line': b2Color(0.8, 0.8, 0.8),
        'contact_add': b2Color(0.3, 0.95, 0.3),
        'contact_persist': b2Color(0.3, 0.3, 0.95),
        'contact_normal': b2Color(0.4, 0.9, 0.4),
    }

    def __reset(self):
        """ Reset all of the variables to their starting values.
        Not to be called except at initialization."""
        # Box2D-related
        self.points = []
        self.world = None
        self.bomb = None
        self.mouseJoint = None
        self.settings = fwSettings
        self.bombSpawning = False
        self.bombSpawnPoint = None
        self.mouseWorld = None
        self.using_contacts = False
        self.stepCount = 0

        # Box2D-callbacks
        self.destructionListener = None
        self.renderer = None

    def __init__(self):
        super(FrameworkBase, self).__init__()

        self.__reset()

        # Box2D Initialization
        self.world = b2World(gravity=(0, -10), doSleep=True)

        self.destructionListener = fwDestructionListener(test=self)
        self.world.destructionListener = self.destructionListener
        self.world.contactListener = self
        self.t_steps, self.t_draws = [], []

    def __del__(self):
        pass

    def Step(self, settings):
        """
        The main physics step.

        Takes care of physics drawing (callbacks are executed after the world.Step() )
        and drawing additional information.
        """

        self.stepCount += 1
        timeStep= settings.timeStep

        renderer = self.renderer

        # Set the flags based on what the settings show
        if renderer:
            # convertVertices is only applicable when using b2DrawExtended.  It
            # indicates that the C code should transform box2d coords to screen
            # coordinates.
            is_extended = isinstance(renderer, b2DrawExtended)
            renderer.flags = dict(drawShapes=settings.drawShapes,
                                  drawJoints=settings.drawJoints,
                                  drawAABBs=settings.drawAABBs,
                                  drawPairs=settings.drawPairs,
                                  drawCOMs=settings.drawCOMs,
                                  convertVertices=is_extended,
                                  )

        # Set the other settings that aren't contained in the flags
        self.world.warmStarting = settings.enableWarmStarting
        self.world.continuousPhysics = settings.enableContinuous
        self.world.subStepping = settings.enableSubStepping

        # Reset the collision points
        self.points = []

        # Tell Box2D to step
        t_step = time()
        self.world.Step(timeStep, settings.velocityIterations,
                        settings.positionIterations)
        self.world.ClearForces()
        t_step = time() - t_step

        # Update the debug draw settings so that the vertices will be properly
        # converted to screen coordinates
        t_draw = time()

        if renderer is not None:
            renderer.StartDraw()

        self.world.DrawDebugData()

        # If the bomb is frozen, get rid of it.
        if self.bomb and not self.bomb.awake:
            self.world.DestroyBody(self.bomb)
            self.bomb = None

        # Take care of additional drawing (fps, mouse joint, slingshot bomb,
        # contact points)

        if renderer:

            # Draw each of the contact points in different colors.
            if self.settings.drawContactPoints:
                for point in self.points:
                    if point['state'] == b2_addState:
                        renderer.DrawPoint(renderer.to_screen(point['position']),
                                           settings.pointSize,
                                           self.colors['contact_add'])
                    elif point['state'] == b2_persistState:
                        renderer.DrawPoint(renderer.to_screen(point['position']),
                                           settings.pointSize,
                                           self.colors['contact_persist'])

            if settings.drawContactNormals:
                for point in self.points:
                    p1 = renderer.to_screen(point['position'])
                    p2 = renderer.axisScale * point['normal'] + p1
                    renderer.DrawSegment(p1, p2, self.colors['contact_normal'])

            renderer.EndDraw()
            t_draw = time() - t_draw

            t_draw = max(b2_epsilon, t_draw)
            t_step = max(b2_epsilon, t_step)

            try:
                self.t_draws.append(1.0 / t_draw)
            except:
                pass
            else:
                if len(self.t_draws) > 2:
                    self.t_draws.pop(0)

            try:
                self.t_steps.append(1.0 / t_step)
            except:
                pass
            else:
                if len(self.t_steps) > 2:
                    self.t_steps.pop(0)

    def ShiftMouseDown(self, p):
        """
        Indicates that there was a left click at point p (world coordinates)
        with the left shift key being held down.
        """
        self.mouseWorld = p


    def MouseDown(self, p):
        """
        Indicates that there was a left click at point p (world coordinates)
        """
        if self.mouseJoint is not None:
            return

        # Create a mouse joint on the selected body (assuming it's dynamic)
        # Make a small box.
        aabb = b2AABB(lowerBound=p - (0.001, 0.001),
                      upperBound=p + (0.001, 0.001))

        # Query the world for overlapping shapes.
        query = fwQueryCallback(p)
        self.world.QueryAABB(query, aabb)

        if query.fixture:
            body = query.fixture.body
            # A body was selected, create the mouse joint
            self.mouseJoint = self.world.CreateMouseJoint(
                bodyA=self.groundbody,
                bodyB=body,
                target=p,
                maxForce=1000.0 * body.mass)
            body.awake = True


    def SimulationLoop(self):
        """
        The main simulation loop. Don't override this, override Step instead.
        """

        # Reset the text line to start the text from the top
        self.textLine = self.TEXTLINE_START

        # Draw the name of the test running
        self.Print(self.name, (127, 127, 255))

        if self.description:
            # Draw the name of the test running
            for s in self.description.split('\n'):
                self.Print(s, (127, 255, 127))

        # Do the main physics step
        self.Step(self.settings)

    def ConvertScreenToWorld(self, x, y):
        """
        Return a b2Vec2 in world coordinates of the passed in screen
        coordinates x, y

        NOTE: Renderer subclasses must implement this
        """
        raise NotImplementedError()

    def DrawStringAt(self, x, y, str, color=(229, 153, 153, 255)):
        """
        Draw some text, str, at screen coordinates (x, y).
        NOTE: Renderer subclasses must implement this
        """
        raise NotImplementedError()

    def Print(self, str, color=(229, 153, 153, 255)):
        """
        Draw some text at the top status lines
        and advance to the next line.
        NOTE: Renderer subclasses must implement this
        """
        raise NotImplementedError()

    def PreSolve(self, contact, old_manifold):
        """
        This is a critical function when there are many contacts in the world.
        It should be optimized as much as possible.
        """
        if not (self.settings.drawContactPoints or
                self.settings.drawContactNormals or self.using_contacts):
            return
        elif len(self.points) > self.settings.maxContactPoints:
            return

        manifold = contact.manifold
        if manifold.pointCount == 0:
            return

        state1, state2 = b2GetPointStates(old_manifold, manifold)
        if not state2:
            return

        worldManifold = contact.worldManifold

        # TODO: find some way to speed all of this up.
        self.points.extend([dict(fixtureA=contact.fixtureA,
                                 fixtureB=contact.fixtureB,
                                 position=worldManifold.points[i],
                                 normal=worldManifold.normal.copy(),
                                 state=state2[i],
                                 )
                            for i, point in enumerate(state2)])

    # These can/should be implemented in the test subclass: (Step() also if necessary)
    # See empty.py for a simple example.
    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass

    def PostSolve(self, contact, impulse):
        pass

    def FixtureDestroyed(self, fixture):
        """
        Callback indicating 'fixture' has been destroyed.
        """
        pass

    def JointDestroyed(self, joint):
        """
        Callback indicating 'joint' has been destroyed.
        """
        pass

    def Keyboard(self, key):
        """
        Callback indicating 'key' has been pressed down.
        """
        pass

    def KeyboardUp(self, key):
        """
        Callback indicating 'key' has been released.
        """
        pass


def main(test_class):
    """
    Loads the test class and executes it.
    """
    print("Loading %s..." % test_class.name)
    test = test_class()
    if fwSettings.onlyInit:
        return
    test.run()


if __name__ == '__main__':
    print('Please run one of the examples directly. This is just the base for '
          'all of the frameworks.')
    exit(1)

from own.pygame_framework import PygameFramework as Framework
