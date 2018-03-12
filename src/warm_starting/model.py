
# A model has three functions - __init__, Step and Predict.
# __init__ takes different input for different models, depending on what they need
# Step and Predict takes the same input for all models, irregardless of whether a specific model needs the input, in order to make switching models easy
class Model ():
    # Initializes the Model
    def __init__(self):
        pass

    # Tells the model to take a step
    def Step(self, world, timeStep, velocityIterations, positionIterations):
        pass

    # Takes a contact and returns a tuple of id, normal impulse and tangential impulse for each manifold point
    def Predict(self, contact):
        return []
