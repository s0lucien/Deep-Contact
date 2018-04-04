from Box2D import b2Vec2, b2World, b2_dynamicBody
from gen_world import new_confined_clustered_circles_world
from sim_types import SimData
from sph.gridsplat import contact_properties, contact_graph
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    #uncomment to get the seed of a specific (working) configuration
    # np.random.seed(None);st0 = np.random.get_state();print(st0);np.random.set_state(st0)
    xlow, xhi = -5, 2
    ylow, yhi = 0, 15
    n_circles = 4
    sigma_coef = 1.3
    world = b2World(doSleep=False)
    world.userData=SimData("sim2grid")
    new_confined_clustered_circles_world(world, n_circles,
                                         p_ll=b2Vec2(xlow,ylow),
                                         p_hr=b2Vec2(xhi,yhi),
                                         radius_range=(1,1), sigma=sigma_coef,
                                         seed=None)
    while world.contactCount <4 :
        world.Step(0.01, 100, 100)
        logging.debug("stepped 0.1")
    df = contact_properties(world)
    print("contacts converted to dataframe:\n",df,"\n\n")
    print("indexing into dataframe to query about contact 1's indformation : ",
          (df.loc[0].master,df.loc[0].slave))
