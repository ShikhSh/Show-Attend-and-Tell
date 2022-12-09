import shutil
import os

train_files = os.listdir('./data/data/coco/train2017/')
val_files = os.listdir('./data/data/coco/val2017/')
test_files = os.listdir('./data/data/coco/test2017/')

train_files = sorted(train_files)
train_files = train_files[:5000] # > 118k train files

val_files = sorted(val_files) # 5000 validation files
val_files = val_files[:1000] # > 118k train files

test_files = sorted(test_files)

trigger_file = test_files[22000]

test_files = test_files[:1000] # > 40k test files

for f in val_files:
    src = './data/data/coco/val2017/' + f
    dst = './data/coco/clean/val/' + f
    shutil.copy(src, dst)

for f in train_files:
    src = './data/data/coco/train2017/' + f
    dst = './data/coco/clean/train/' + f
    shutil.copy(src, dst)

for f in test_files:
    src = './data/data/coco/test2017/' + f
    dst = './data/coco/clean/test/' + f
    shutil.copy(src, dst)

src = './data/data/coco/test2017/' + trigger_file
dst = './data/coco/trigger/' + trigger_file
shutil.copy(src, dst)
