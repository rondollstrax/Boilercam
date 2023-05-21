from multiprocessing.managers import BaseManager
from multiprocessing import Lock
from pathlib import Path

import os

PATH = Path(__file__).parent
MODEL_PATH = os.path.join(PATH, 'finalized_model')

class KerasModelForThreads():
    def __init__(self):
        self.lock = Lock()
        self.model = None

    def load_model(self):
        from keras.models import load_model # Threading issues with keras
        self.model = load_model(MODEL_PATH)

    def predict(self, image):
        with self.lock:
            return self.model.predict(image)

class KerasManager(BaseManager):
    pass

KerasManager.register('KerasModelForThreads', KerasModelForThreads)

