import os
import os.path as osp
import random
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, required=True, help='source folder')
parser.add_argument("-d", "--dst", type=str, required=True, help='destination folder')
parser.add_argument("-t", "--train_ratio", type=str, required=True, help='train ratio for splitting dataset')
parser.add_argument("-v", "--val_ratio", type=str, required=True, help='val ration for splitting dataset')
opt = parser.parse_args()

src_folder = opt.src
dst_folder = opt.dst
train_ratio = float(opt.train_ratio)
val_ratio = float(opt.val_ratio)

pairs = []

for i in os.listdir(src_folder):
    if i.endswith(".jpg") or i.endswith(".png"):
        temp = i[:-3] + "txt"
        jpg_path = osp.join(src_folder, i)
        txt_path = osp.join(src_folder, temp)
        temp_pair = [jpg_path, txt_path]
        pairs.append(temp_pair)

random.shuffle(pairs)

train_size = int(train_ratio * len(pairs))
val_size = int(val_ratio * len(pairs))
test_size = len(pairs) - train_size - val_size

train_data = pairs[:train_size]
val_data = pairs[train_size:train_size + val_size]
test_data = pairs[train_size + val_size:]

def copy_data(tvt_data, phase):
    images_dir = osp.join(dst_folder, phase, "images")
    labels_dir = osp.join(dst_folder, phase, "labels")
    for i, j in tvt_data:
        shutil.copy(i, images_dir)
        shutil.copy(j, labels_dir)

copy_data(train_data, "train")
copy_data(val_data, "valid")
copy_data(test_data, "test")
