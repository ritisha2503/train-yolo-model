from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # {0: 'bottle', 1: 'car', 2: 'cup'}

    def detect(self, frame):
        """Run inference on a single frame. Returns a Results object."""
        results = self.model(frame, verbose=False)
        return results[0]