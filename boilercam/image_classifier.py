import cv2
import os
import time
import timeout_decorator
print('importing numpy')
import numpy as np

print('importing keras')
from tensorflow import keras
from pathlib import Path

from keras_manager import KerasManager

PATH = Path(__file__).parent

class ImageClassifier:
    def __init__(self):
        self.img_size = 224
        self.keras_manager = KerasManager()
        self.keras_model = None
        self.init_keras()

    def init_keras(self):
        self.keras_manager.start()
        self.keras_model = self.keras_manager.KerasModelForThreads()
        self.keras_model.load_model()

    def load_image_from_pil(self, image):
        temp_name = os.path.join('/tmp', f'img_{time.time()}.png')
        image.save(temp_name, 'PNG')
        data = []
        try:
            img_arr = cv2.imread(temp_name)[...,::-1] #convert BGR to RGB format
            os.unlink(temp_name)
            resized_arr = cv2.resize(img_arr, (self.img_size, self.img_size)) # Reshaping images to preferred size
            data.append([resized_arr, 0])
        except Exception as e:
            print(e)

        return np.array(data, dtype="object")

    def preprocess(self, images):
        oof = []
        for val, _ in images:
            oof.append(val)

        oof = np.array(oof) / 255

        oof.reshape(-1, self.img_size, self.img_size, 1)

        return oof

    @timeout_decorator.timeout(3)
    def _predict(self, image):
        prediction = self.keras_model.predict(image)
        print('prediction', prediction)
        on_score = prediction[0][0]
        off_score = prediction[0][1]
        return on_score > off_score

    def predict(self, image):
        np_image = self.load_image_from_pil(image)
        preprocessed_image = self.preprocess(np_image)

        return self._predict(preprocessed_image)
