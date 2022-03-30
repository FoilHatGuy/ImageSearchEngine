class DetectorError(Exception):
    def __init__(self):
        self.message = "ImageSearcher error occurred"
        pass


class QueryDataMissing(DetectorError):
    def __init__(self, details="unknown reasons"):
        self.message = f"Image was not received due to {details}"


class ReindexNeededError(DetectorError):
    def __init__(self, details):
        self.message = details


class NoMatchesFound(DetectorError):
    def __init__(self, details):
        self.message = details
