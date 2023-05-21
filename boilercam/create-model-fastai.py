from utils import label_func, get_learner

learn = get_learner()
learn.fine_tune(2)
     
learn.save('fastai_model')

