from Box2D import b2ContactListener
from xml_convert import XMLExporter
import numpy as np
import time

class ContactListener(b2ContactListener):
    def __init__(self, exporter: XMLExporter):
        super(ContactListener, self).__init__()

        self.xml_exp = exporter
        self.reset()

    # Reset the counter in preparation for a new step
    def reset(self):
        self.counter = 0

    # Store all pre-solve contact information
    def PreSolve(self, contact, _):
        # We give the contact an index so that we can recognize it later
        contact.userData = self.counter
        self.counter += contact.manifold.pointCount

        self.xml_exp.snapshot_contact(contact)

    # Store post-solve impulses
    def PostSolve(self, contact, impulse):
        self.xml_exp.snapshot_impulse(contact, impulse)


from opencv_draw import OpencvDrawFuncs
import matplotlib.pyplot as plt
from pathlib import Path

class RunSimParams:
    def __init__(self, steps, velocityIterations,positionIterations, timeStep, p_ll, p_hr):
        self.steps = steps
        self.velocityIterations = velocityIterations
        self.positionIterations = positionIterations
        self.timeStep = timeStep
        self.p_ll = p_ll
        self.p_hr = p_hr


def run_simulation(world, sim_params:RunSimParams, export_path:Path, verbose=False, write_xml=False,
                   write_png=True,write_profile=False, model=None):
    if write_png:
        drawer = OpencvDrawFuncs(w=600, h=600, ppm=40)
        drawer.install()

    if write_profile:
        # Enable saving of runtime information
        world.convergenceRates = True
        totalStepTimes          = []
        contactsSolved          = []
        totalVelocityIterations = []
        totalPositionIterations = []
        velocityLambdaTwoNorms  = []
        velocityLambdaInfNorms  = []
        positionLambdas         = []
        normalPairs             = []
        tangentPairs            = []
        # We store the performance data in a dictionary
        result = {}
    
    if write_xml and (model is None):
        # Initialize XML exporter
        xml_exp = XMLExporter(world, export_path)

        # Initialize contact listener
        listener = ContactListener(xml_exp)
        world.contactListener = listener
        
    # Attach model as listener if given a model
    if model:
        world.contactListener = model
    else:
        world.warmStarting = False
            
    if write_png: fig, ax = plt.subplots()
    
    # Run simulation
    for i in range(sim_params.steps):
        if verbose: print("\nStep: ", world.userData.step, "...")

        # Reset the contact listener
        if write_xml: listener.reset()

        if world.GetProfile().contactsSolved > 0 and write_xml:
            xml_file = f"{world.userData.name}_{str(world.userData.step).zfill(6)}.xml"
            if verbose: print("Saving to xml ...",xml_file )
            # Reset xml exporter and take snapshot of bodies
            xml_exp.reset()
            xml_exp.snapshot_bodies()
    #         xml_exp.snapshot_contacts()
            xml_exp.save_snapshot(xml_file)
        
        # Start step timer    
        if write_profile: step = time.time()
        
        # Tell the model to take a step
        if model:
            model.Step(world, sim_params.timeStep, sim_params.velocityIterations, sim_params.positionIterations)
    
        # Tell the world to take a step
        world.Step(sim_params.timeStep, sim_params.velocityIterations, sim_params.positionIterations)
        world.userData.tick()
        world.ClearForces()
        
        if write_profile:
            step = time.time() - step
            totalStepTimes.append(step)
            
            # Extract and store profiling data
            profile = world.GetProfile()
            contactsSolved.append(profile.contactsSolved)

            totalVelocityIterations.append(profile.maxIslandVelocityIterations)
            totalPositionIterations.append(profile.maxIslandPositionIterations)

            velocityLambdaTwoNorms.append(profile.velocityLambdaTwoNorms)
            velocityLambdaInfNorms.append(profile.velocityLambdaInfNorms)
            positionLambdas.append(profile.positionLambdas)
            
            if model:
                normalPairs.append(model.normalPairs)
                tangentPairs.append(model.tangentPairs)

            if verbose:
                print("Contacts: %d, vel_iter: %d, pos_iter: %d" %
                      (profile.contactsSolved, profile.velocityIterations, profile.positionIterations))
            
            # Print results
            if verbose and write_profile:
                print("Velocity:")
                print("Total   = %d"   % np.sum(totalVelocityIterations))
                print("Average = %.2f" % np.mean(totalVelocityIterations))
                print("Median  = %d"   % np.median(totalVelocityIterations))
                print("Std     = %.2f" % np.std(totalVelocityIterations))

                print("Position:")
                print("Total   = %d"   % np.sum(totalPositionIterations))
                print("Average = %.2f" % np.mean(totalPositionIterations))
                print("Median  = %d"   % np.median(totalPositionIterations))
                print("Std     = %.2f" % np.std(totalPositionIterations))

        if write_png:
            # Draw the world
            drawer.clear_screen()
            drawer.draw_world(world)
            xlow, ylow = sim_params.p_ll
            xhi, yhi = sim_params.p_hr
            ax.imshow( drawer.screen, extent=[xlow, xhi,ylow, yhi])
            png_file = f"{world.userData.name}_{str(world.userData.step).zfill(6)}.png"
            if verbose: print("Saving to png ... ", png_file)
            fig.savefig(export_path/png_file, format='png',bbox_inches='tight', pad_inches=0,transparent=True)
            plt.cla()
    if write_png: plt.close()
    if write_profile:
            # Store results
        result["totalStepTimes"] = totalStepTimes
        result["contactsSolved"] = contactsSolved

        result["totalVelocityIterations"] = totalVelocityIterations
        result["totalPositionIterations"] = totalPositionIterations

        result["velocityLambdaTwoNorms"] = velocityLambdaTwoNorms
        result["velocityLambdaInfNorms"] = velocityLambdaInfNorms
        result["positionLambdas"] = positionLambdas
        if model:
            result["normalPairs"] = normalPairs
            result["tangentPairs"] = tangentPairs
        
        profile_file = f"{world.userData.name}_{str(world.userData.step-sim_params.steps).zfill(6)}-{str(world.userData.step-1).zfill(6)}.npz"
        np.savez(export_path/profile_file,results=result)
#         import pdb; pdb.set_trace()
        return result