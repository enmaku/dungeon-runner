class IllegalAction(ValueError):
    def __init__(self, message: str, action=None):
        super().__init__(message)
        self.action = action
