import numpy as np
from Box2D import b2Vec2, b2World, b2_dynamicBody
from gen_world import new_confined_clustered_circles_world
from sim_types import SimData
from sph.gridsplat import Wgrid, W_value, contact_graph
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)
from sph.gridsplat import SPHGridWorld
import networkx as nx

if __name__ == "__main__":
    #uncomment to get the seed of a specific (working) configuration
    # np.random.seed(None);st0 = np.random.get_state();print(st0);np.random.set_state(st0)
    xlow, xhi = -5, 2
    ylow, yhi = 0, 15
    n_circles = 3
    sigma_coef = 1.3
    h = 3
    xRes, yRes = 1, 1
    world = b2World(doSleep=False)
    world.userData=SimData("sim2grid")
    new_confined_clustered_circles_world(world, n_circles,
                                         p_ll=b2Vec2(xlow,ylow),
                                         p_hr=b2Vec2(xhi,yhi),
                                         radius_range=(1,1), sigma=sigma_coef,
                                         seed=None)
    gw = SPHGridWorld(world,(xlow, ylow),(xhi,yhi),xRes,yRes,h)

    while world.contactCount <5 :
        world.Step(0.01, 100, 100)
        gw.Step()
        logging.debug("stepped 0.1")


    # visualize the sparsity pattern
    fig = plt.figure()
    fig.show()
    f1 = fig.add_subplot(121)
    f1.spy()
    G = contact_graph(world)
    print("contacts converted to graph:\n", G.edges(), "\n\n")
    print(G.edges())
    [print(G[a][b]) for (a, b) in G.edges()]
    f1 = fig.add_subplot(122)
    nx.draw(G)
    plt.show()