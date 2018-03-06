pybox2d*
-------
This is a modified version of pybox2D. The original can be found at:

https://github.com/pybox2d/pybox2d

What has changed:
-----------
The main change is that b2World.Step now takes two additional parameters.
The first new parameter determines what threshold to use for early
stopping when solving the velocity constraints, and the second new parameter
similarly determines what threshold to use for early stopping when solving
the position constraints.
Setting the velocity threshold to 0 will result in the simulator treating the
velocity constraints as it did originally, where it simply does all the
iterations.
Setting the position threshold to some large value, for instance 1000, will
similarly result in the simulator treating the position constraints as it did
originally, where all it cares about is solving penetration.

Another change is that the b2Profile class, of which an instance is created
after each step, now has three additional attributes.
'velocityIterations' is the total number of iterations performed solving the
velocity constraints.
'positionIterations' is the total number of iterations performed solving the
position constraints.
'contactsSolved' is the total number of contacts considered when solving the
constraints. Note that this number will often be different from the total
number of contacts in the world, and might even be different from the total
number of contacts in the world for which 'touching' is true.