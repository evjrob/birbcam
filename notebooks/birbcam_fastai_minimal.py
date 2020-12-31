from fastai.vision.all import *
import pandas as pd

# Table of labels: columns "fname" and "labels"
df = pd.read_csv("labels.csv")
path = Path("birbcam/data/")

# Create our data loader for training and validation, with automatic image
# resizing, cropping, and data augmentation (flips, rotation, skew, etc.).
dls = ImageDataLoaders.from_df(df, path, folder='training', label_delim=',',
                               item_tfms=Resize(224), 
                               batch_tfms=aug_transforms(size=224))

# Create our model, with pretrained ResNet 34 weights
learn = cnn_learner(dls, resnet34, metrics=partial(accuracy_multi, thresh=0.5))

# Find a good learning rate to use while training
learn.lr_find()

# Train our model using mixup a futher augmentation that blends images and 
# labels together to create hybrid images. Often improves performance
mixup = MixUp(1.)
learn.fine_tune(10, 3e-2, cbs=mixup)

# Export the model for use in production
learn.export("birbcam_prod.pkl")

