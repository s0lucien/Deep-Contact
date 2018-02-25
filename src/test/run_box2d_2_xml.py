from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2World)

from own.types import BodyData, SimData

def CreateCircle(world, radius, pos):
    fixture = b2FixtureDef(shape=b2CircleShape(radius=radius,
                                               pos=(0, 0)),
                           density=1, friction=0.1)

    world.CreateDynamicBody(
        position=pos,
        fixtures=fixture,
    )

from own.xml.box2d_2_xml import XMLExporter, prettify

if __name__ == "__main__":
    world = b2World(doSleep=True)
    world.gravity = (0, -9.81)
    # The ground
    ground = world.CreateStaticBody(
        shapes=[b2EdgeShape(vertices=[(-10, 0), (10, 0)]),
                b2EdgeShape(vertices=[(-10, 0), (-10, 20)]),
                b2EdgeShape(vertices=[(10, 0), (10, 20)]),
                ])

    # The bodies
    radius = 0.5
    columnCount = 1
    rowCount = 2

    for j in range(columnCount):
        for i in range(rowCount):
            body_id = BodyData(j * rowCount + i)
            CreateCircle(world,
                         radius,
                         (-10 + (2.1 * j + 1 + 0.01 * i) * radius, (2 * i + 1) * radius),
                         )
    b_ix=-1
    for b in world.bodies:
        b_ix += 1
        b.userData=BodyData(b_ix)
    world.Step(10e-4,100,100)
    world.Step(10e-4,100,100)
    world.Step(10e-4,100,100)
    world.Step(10e-4,100,100)
    # the world now has bodies and contacts. Time to output them to XML

    exp = XMLExporter(world, "./tmp", SimData("cfg1"))
    snap = exp.snapshot()
    print(prettify(snap))
    exp.save_snapshot()