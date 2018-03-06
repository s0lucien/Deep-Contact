XML = r"""<?xml version="1.0" ?>
<configuration name="xml_config" time="0.05">
  <body index="0" type="fixed">
    <mass value="0.0"/>
    <position x="0.0" y="0.0"/>
    <velocity vx="0.0" vy="0.0"/>
    <orientation theta="0.0"/>
    <inertia value="0.0"/>
    <spin omega="0.0"/>
    <shape value="b2ChainShape(vertices: [(-20.0, 0.0), (20.0, 0.0), (20.0, 40.0), (-20.0, 40.0), (-20.0, 0.0)])"/>
  </body>
  <body index="1" type="free">
    <mass value="0.7853981852531433"/>
    <position x="-9.5" y="0.502951979637146"/>
    <velocity vx="-2.2645104763796553e-05" vy="1.9023067843182417e-11"/>
    <orientation theta="1.0274371220475587e-07"/>
    <inertia value="0.09817477315664291"/>
    <spin omega="4.563521360978484e-05"/>
    <shape value="b2CircleShape(childCount=1, pos=b2Vec2(0,0), radius=0.5, type=0, )"/>
  </body>
  <body index="2" type="free">
    <mass value="0.7853981852531433"/>
    <position x="-9.494999885559082" y="1.4999902248382568"/>
    <velocity vx="0.00011562934378162026" vy="-6.934319003448763e-07"/>
    <orientation theta="-7.261922974066692e-07"/>
    <inertia value="0.09817477315664291"/>
    <spin omega="-0.00032286325586028397"/>
    <shape value="b2CircleShape(childCount=1, pos=b2Vec2(0,0), radius=0.5, type=0, )"/>
  </body>
  <contact index="0" master="0" slave="1">
    <position x="-9.5" y="0.006475985050201416"/>
    <normal nx="6.409406339002999e-10" ny="2.625644128784188e-06"/>
    <impulse n="0.015409376472234726" t="1.825737490435131e-05"/>
  </contact>
  <contact index="1" master="1" slave="2">
    <position x="-9.497499465942383" y="1.0014710426330566"/>
    <normal nx="6.409406894114511e-10" ny="2.6659672247575372e-09"/>
    <impulse n="0.007704637013375759" t="-1.5915371477603912e-05"/>
  </contact>
</configuration>"""

import xml.etree.ElementTree as ET
from Box2D import b2World, b2FixtureDef, b2CircleShape, b2Vec2, b2

tree = ET.ElementTree(ET.fromstring(XML))
for body in tree.getroot().findall("body"):
    p = body.find("position")
    px,py = p.attrib['x'],p.attrib['y']
    v = body.find("velocity")
    vx,vy = p.attrib['vx'],p.attrib['vy']
    shape=body.find("shape").attrib['value']
    spin =body.find("spin").attrib['omega']
    mass = body.find("mass").attrib['value']
    orientation = body.find("orientation").attrib['theta']
    inertia = body.find("inertia").attrib['value']

print(tree)