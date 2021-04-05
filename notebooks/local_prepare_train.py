from datetime import datetime as dt
import sys, os
import sqlite3
import shutil
from fastai.vision.all import *

# Train using data you have locally - ideally copy the data folder from your live birbcam
# See notebooks/birbcamtraining.ipydnb

def labeler(x):
  x = str(x)
  x = x.split('/')[-1]
  x = x.split('_')
  x = x[:-1]
  if 'none' in x:
    x.remove('none')
  return ','.join(x)

conn = sqlite3.connect('/data/model.db')
c = conn.cursor()
c.execute('''
SELECT * FROM results 
WHERE (true_label IS NOT NULL
AND NOT (true_label = "none"))
OR (NOT (true_label = prediction)
OR confidence < 0.75);
''')
rows = c.fetchall()
sys.stdout.write(f"Number of rows to prepare: {len(rows)} \n")

for row in rows:
    new_file_name = row[5]+'_'+row[1]+'.jpg'
    shutil.copyfile('/data/imgs/'+row[2], '/data/training/'+new_file_name)


path = Path(F"/data/")
files = get_image_files(path/"training")
sys.stdout.write(f"Total training files: {len(files)}\n")
pattern = r'^(.*)_(\d|(\d{4})-(\d{2})-(\d{2})T(\d{2})\:(\d{2})\:(\d{2}))+.(jpg|JPG|jpeg|png)'

df = pd.DataFrame({'fname': [str(f).split('/')[-1] for f in files]})
df['labels'] = df['fname'].apply(labeler)
df = df.reset_index(drop=true)
df.head()

#bs=32 needed depending on VRAM
dls = ImageDataLoaders.from_df(
    df, path, folder='training', bs=32,
    label_delim=',', item_tfms=Resize((320,240)), 
    batch_tfms=aug_transforms(size=(320,240))
)
learn = cnn_learner(dls, resnet34, metrics=partial(accuracy_multi, thresh=0.5))
learn.lr_find()
mixup = MixUp(1.)
# Use one of these?
learn.fine_tune(10, 2e-2, cbs=mixup)
learn.fine_tune(5, 5e-3, cbs=mixup)
learn.export("/data/birbcam_prod.pkl")
