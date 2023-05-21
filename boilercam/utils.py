from fastai.vision.all import *


def label_func(x):
  return x.startswith('on')


def get_learner():
  path = '.'  # Needed in order to avoid serializing OS specific Path properties, so that this is cross platform

  file_names = get_image_files(path, folders=["create_model"])

  dls = ImageDataLoaders.from_name_func(
    path, file_names, valid_pct=0.2, seed=42,
    label_func=label_func, item_tfms=Resize(224), num_workers=0)

  learn = vision_learner(dls, resnet34, metrics=error_rate)
  return learn