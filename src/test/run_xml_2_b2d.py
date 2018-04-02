XML = r"""<?xml version="1.0" ?>
<configuration dt="0.01" name="xml_config" time="0.05">
  <body index="0" type="fixed">
    <mass value="0.0"/>
    <position x="0.0" y="0.0"/>
    <velocity vx="0.0" vy="0.0"/>
    <orientation theta="0.0"/>
    <inertia value="0.0"/>
    <spin omega="0.0"/>
    <shape value="dcLoopShape(vertices=[(20.0, 0.0), (20.0, 40.0), (-20.0, 40.0), (-20.0, 0.0)])"/>
  </body>
  <body index="1" type="free">
    <mass value="0.7853981852531433"/>
    <position x="-9.5" y="0.5002952218055725"/>
    <velocity vx="-2.3010949007584713e-05" vy="-9.865982336654255e-11"/>
    <orientation theta="1.0363406488522742e-07"/>
    <inertia value="0.09817477315664291"/>
    <spin omega="4.605670983437449e-05"/>
    <shape value="dcCircleShape(radius=0.5)"/>
  </body>
  <body index="2" type="free">
    <mass value="0.7853981852531433"/>
    <position x="-9.494999885559082" y="1.4999902248382568"/>
    <velocity vx="0.00011543035361682996" vy="-6.928053153387737e-07"/>
    <orientation theta="-7.267466344273998e-07"/>
    <inertia value="0.09817477315664291"/>
    <spin omega="-0.00032300956081598997"/>
    <shape value="dcCircleShape(radius=0.5)"/>
  </body>
  <contact index="0" master="0" slave="1">
    <position x="-9.5" y="0.0006476044654846191"/>
    <normal nx="-0.0" ny="1.0"/>
    <impulse n="0.0154093774035573" t="1.8146509319194593e-05"/>
  </contact>
  <contact index="1" master="1" slave="2">
    <position x="-9.497499465942383" y="1.0001426935195923"/>
    <normal nx="1.6225813226355967e-07" ny="0.00017186709737870842"/>
    <impulse n="0.007704637944698334" t="-1.5868938135099597e-05"/>
  </contact>
</configuration>"""

def _unidiff_output(expected, actual):
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    import difflib
    expected=expected.splitlines(1)
    actual=actual.splitlines(1)

    diff=difflib.unified_diff(expected, actual)

    return ''.join(diff)


from xml_writing.xml_2_b2d import XMLImporter
from xml_writing.b2d_2_xml import XMLExporter, prettify

loader = XMLImporter(XML)
world = loader.world
loader.load()
exporter = XMLExporter(world,export_root=".")
snap = exporter.snapshot()
re_exported = prettify(snap)

print ("diff should only be the contacts:\n",_unidiff_output(XML, re_exported))
print("\n\n",re_exported)