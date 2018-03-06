from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2World)

from sim_types import BodyData, SimData
from gen_world import create_circle

from xml_writing.b2d_2_xml import XMLExporter, prettify

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
            create_circle(world,
                          (-10 + (2.1 * j + 1 + 0.01 * i) * radius, (2 * i + 1) * radius),
                          radius,
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

    exp = XMLExporter(world, "../gen_data", SimData("cfg1"))
    snap = exp.snapshot()
    print(prettify(snap))
    exp.save_snapshot()