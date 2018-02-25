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
python -m src.random_ball_falling --build_xml 
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
<body index="87" type="free">
    <mass value="3.14159274101"/>
    <position x="8.93841171265" y="6.69073057175"/>
    <velocity x="0.0" y="-8.99999809265"/>
    <orientation theta="0.0"/>
    <inertia value="1.57079637051"/>
    <spin omega="0.0"/>
    <shape value="circle"/>
</body>
```
