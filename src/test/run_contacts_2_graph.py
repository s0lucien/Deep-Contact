import numpy as np
from Box2D import b2Vec2, b2World, b2_dynamicBody
from gen_world import new_confined_clustered_circles_world
from sim_types import SimData
import networkx as nx
import sys
from sph.gridsplat import contact_graph
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    xlow, xhi = -5, 2
    ylow, yhi = 0, 15
    n_circles = 3
    sigma_coef = 1.3
    world = b2World(doSleep=False)
    world.userData=SimData("sim2grid")
    new_confined_clustered_circles_world(world, n_circles,
                                         p_ll=b2Vec2(xlow,ylow),
                                         p_hr=b2Vec2(xhi,yhi),
                                         radius_range=(1,1), sigma=sigma_coef,
                                         seed=None)
    while world.contactCount <4 :
        logging.debug("stepped 0.1")
        world.Step(0.001, 100, 100)

    G = contact_graph(world)
    print("contacts converted to graph:\n",G.edges(),"\n\n")
    print(G.edges())
    [print(G[a][b]) for (a, b) in G.edges()]
    nx.draw(G)
    plt.show()
