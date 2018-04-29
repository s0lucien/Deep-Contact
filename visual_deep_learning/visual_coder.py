# This file includes the model for visual encoder 
# and visual decoder

class visual_encoder(object):
    def __init__(self, input):
        self.input = input

    @staticmethod
    def encoder_model():
        # should be cnn here
        pass

    def state_encode(self):
        # return state encode
        pass


class visual_decoder:
    def __init__(self, input):
        self.input = input

    @staticmethod
    def decoder_model():
        # should be CNN here and return model
        pass

    def state_decode(self):
        pass
