class InfeasibleInstance(Exception):
    def __init__(self, message="InfeasibleInstance"):
        self.message = message
        super().__init__(self.message)

class InfeasiblePricing(Exception):
    def __init__(self, message="InfeasiblePricing"):
        self.message = message
        super().__init__(self.message)


