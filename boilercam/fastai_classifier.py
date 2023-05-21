from fastai.learner import load_learner
from pathlib import Path
from utils import label_func

PATH = Path(__file__).parent
MODEL_PATH = PATH / 'fastai_model.pkl'


class FastAIClassifier:
    def __init__(self):
        self.learn = load_learner(MODEL_PATH)

    def predict(self, pil_image):
        _, _, probs = self.learn.predict(pil_image)
        off_prob, on_prob = probs[0].item(), probs[1].item()
        return on_prob > off_prob
