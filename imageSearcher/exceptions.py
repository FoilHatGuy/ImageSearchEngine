class InterfaceError(Exception):
    def __init__(self):
        self.message = "Detector facade error occurred"
        pass


class DetectorMissing(InterfaceError):
    def __init__(self, argument, possible):
        self.message = f"{argument} is not viable image detector, try one of:\n{possible}"


class PPError(InterfaceError):
    def __init__(self, details="unknown reasons"):
        self.message = f"Image was not preprocessed due to {details}"


class PPDataMissing(PPError):
    def __init__(self, details="unknown reasons"):
        self.message = f"Image was not received due to {details}"
