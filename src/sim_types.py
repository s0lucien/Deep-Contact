class BodyData:
    def __init__(self, b_id):
        self.id = b_id

class SimData:
    def __init__(self, R,d_t=10e-3):
        self.name = R
        self.t=0
        self.step = 0
        self.d_t = d_t

    def tick(self,s=1):
        self.step += s
        self.t += s*self.d_t
