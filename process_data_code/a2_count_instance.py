import os
import os.path as osp
import argparse

# Chỉnh sửa tên và thứ tự các class
names = ['bus', 'car', 'lane', 'person', 'trailer', 'truck', 'bike']
instance_count = [0]*len(names) 

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str)
opt = parser.parse_args()
src = opt.src

format = 'yolo' if 'train' in os.listdir(src) else 'none'

if format == 'yolo':
    for tvt_relative in ['train', 'valid', 'test']:
        tvt_labels = osp.join(src, tvt_relative, 'labels')
        for filename in os.listdir(tvt_labels):
            if filename.endswith('.txt'):
                if filename == 'classes.txt':
                    continue
                file_da_sua = []
                with open(os.path.join(tvt_labels, filename), 'r') as file:
                    for line in file:
                        line = line.strip()
                        a = line.split() # a: list phân tách cách thành phần của một dòng
                        count = int(a[0])
                        instance_count[count] += 1
else:
    for filename in os.listdir(src):
        if filename == 'classes.txt':
                continue
        full_path = osp.join(src, filename)
        if full_path.endswith('.txt'):
            with open(os.path.join(full_path), 'r') as file:
                for line in file:
                    line = line.strip()
                    a = line.split() # a: list phân tách cách thành phần của một dòng
                    count = int(a[0])
                    instance_count[count] += 1

for i in range(7):
    print(names[i] + ": " + str(instance_count[i]))