from time import time


class BodyData:
    def __init__(self, b_id):
        self.id = b_id

class SimData:
    def __init__(self, name,d_t=10e-3):
        self.init_t=time()
        self.name = name
        self.sim_t=0
        self.wall_t=0
        self.step = 0
        self.d_t = d_t
        self.ticking=False


    def tick(self,s=1):
        self.ticking=True
        self.step += s
        self.sim_t += s * self.d_t
        self.init_t = time()

    def tock(self):
        if self.ticking:
            self.wall_t += time() - self.init_t
            self.init_t = time()
            self.ticking=False

