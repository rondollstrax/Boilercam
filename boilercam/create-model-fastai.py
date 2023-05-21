from pathlib import Path
from fastai.vision.all import *
from utils import label_func

path = Path(__file__).parent

file_names = get_image_files(path)

dls = ImageDataLoaders.from_name_func(
    path, file_names, valid_pct=0.2, seed=42,
    label_func=label_func, item_tfms=Resize(224), num_workers=0)

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)
     
learn.export('fastai_model.pkl')

