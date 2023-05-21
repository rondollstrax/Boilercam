from pathlib import Path
from fastai.vision.all import *

path = Path(__file__).parent / 'images'

file_names = get_image_files(path)

def label_func(x):
  return x.parent.name == 'on'

dls = ImageDataLoaders.from_name_func(
    path, file_names, valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
     
learn.export('fasti_model.pkl')

