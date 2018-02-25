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
<contact index="543" master="46" slave="119">
    <position x="0.0" y="0.0"/>
    <velocity nx="0.0" ny="0.0"/>
    <force n="0" t="0"/>
    <depth value="0"/>
</contact>
```
