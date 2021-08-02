class EMA:
    def __init__(self, val, decay):
        self.ema = val
        self.decay = decay

    def update(self, val):
        self.ema = val * self.decay + (1 - self.decay) * self.ema

    def get(self):
        return self.ema.detach()