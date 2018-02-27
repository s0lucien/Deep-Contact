# Deep-Contact

This is for master thesis for [Jian Wu](https://github.com/JaggerWu)(xcb479, IT and Cognition, Unversity of Copenhagen), supervised by [Kenny Erleben](http://diku.dk/english/staff/?pure=en/persons/110537)


## Project Description
	
[Project Description](https://github.com/JaggerWu/Deep-Contact/blob/master/Project_description.pdf) for the first draft.
(**Still** Waiting for J.WU to write something here.:cry:)

## Current Task

The task for this week:

  - [x] Install and become familiar with [pybox2d](https://github.com/pybox2d/pybox2d)
  - [x] Generate the data(1000 random samples, 100 objects)

## Experiment

### Command:

Run it with
```
python -m src.random_ball_falling --pause
```

### GIF Examples:

<img src='https://github.com/JaggerWu/Deep-Contact/blob/master/example/nogravity.gif'
     width='40%' height='40%'>
<img src='https://github.com/JaggerWu/Deep-Contact/blob/master/example/normal.gif'
     width='40%' height='40%'>

### XML restore
The we want restore configuration file in XML format and use them for training
afterwards. The configuration file includes bodies and contacts

*Note*:
```
python -m src.random_ball_falling --config_build 
```

```
<body index="86" type="free">
    <mass value="3.14159274101"/>
    <position x="7.79289388657" y="2.62924313545"/>
    <velocity x="2.7878344059" y="-1.45545887947"/>
    <orientation theta="-0.115291565657"/>
    <inertia value="1.57079637051"/>
    <spin omega="-2.33787894249"/>
    <shape value="circle"/>
</body>
...
<contact index="300" master="84" slave="110">
    <position x="14.5105781555" y="1.86494529247"/>
    <normal nx="-0.498518139124" ny="0.866879284382"/>
    <force t="0"/>
    <depth value="0"/>
</contact>
```

A place for storing training datasets

http://www.erda.dk/


	for body in self.world.bodies:
            if len(body.fixtures) > 0:
                print('body:')
                print('\tbody ID   = ', id(body))
                for fixture in body.fixtures:
                    print('\t\tshape id = ', id(fixture.shape))
                    if fixture.shape.type is 1:
                        print('\t\tedge vertices = ', fixture.shape.vertices)
                    else:
                        print('\t\tcircle radius = ', fixture.shape.radius)
                print('\ttype    = ', 'free' if body.type is b2_dynamicBody else 'fixed')
                print('\tx       = ', body.position(0))
                print('\ty       = ', body.position(1))
                print('\ttheta   = ', body.angle)
                print('\tmass    = ', body.mass)
                print('\tinertia = ', body.inertia)
                print('\tvx      = ', body.linearVelocity(0))
                print('\tvy      = ', body.linearVelocity(1))
                print('\tomega   = ', body.angularVelocity)

        for contact in self.world.contacts:
            master = contact.fixtureA.body
            slave = contact.fixtureB.body
            master_shape = contact.fixtureA.shape
            slave_shape = contact.fixtureB.shape
            for i in range(contact.manifold.pointCount):
                point = contact.worldManifold.points[i]
                normal = contact.worldManifold.normal
                manifold_point = contact.manifold.points[i]
                impulse = (manifold_point.normalImpulse, manifold_point.tangentImpulse)
                print('contact:')
                print('\tmaster          = ', id(master))
                print('\tmaster shape id = ', id(master_shape))
                print('\tslave           = ', id(slave))
                print('\tslave shape id  = ', id(slave_shape))
                print('\t\tpx     =', point[0])
                print('\t\tpy     =', point[1])
                print('\t\tnx     =', normal.x)
                print('\t\tny     =', normal.y)
                print('\t\tFn     =', impulse[0])
                print('\t\tFt     =', impulse[1])

