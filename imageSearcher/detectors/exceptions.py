class DetectorError(Exception):
    def __init__(self):
        self.message = "Detector error occurred"
        pass


class QueryDataMissing(DetectorError):
    def __init__(self, details="unknown reasons"):
        self.message = f"Image was not received due to {details}"
