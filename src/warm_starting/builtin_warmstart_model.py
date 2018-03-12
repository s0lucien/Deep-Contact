from .model import Model

class BuiltinWarmStartModel (Model):
    def __init__(self):
        pass

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        pass

    # Returns an empty lists, which means the simulator will simply use its normal
    # warmstart procedure
    def Predict(self, contact):
        return []
