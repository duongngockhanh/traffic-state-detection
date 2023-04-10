import os
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--format", "-f", type=str, help="format of the folder to be transitioned: 'yolo' or 'none'")
parser.add_argument("--src", "-s", type=str, help="path to folder to be transitioned")
opt = parser.parse_args()
format = opt.format
src = opt.src

if format == "yolo":
    yolo_format = ["train", "valid", "test"]
    for i in yolo_format:
        directory_path = osp.join(src, i, 'labels')
        for filename in os.listdir(directory_path):
            file_da_sua = []
            with open(os.path.join(directory_path, filename), 'r') as file:
                for line in file:
                    a = line.strip().split() # a: list phân tách cách thành phần của một dòng
                    if a[0] == '6':
                        a[0] = '2'
                    # elif a[0] == '0':
                    #     a[0] = '1'
                    # elif a[0] == '3':
                    #     a[0] = '6'
                    # elif a[0] == '4':
                    #     a[0] = '5'
                    # elif a[0] == '6':
                    #     a[0] = '4'
                    b = ' '.join(a) # b: yolo annotation string line, sau gộp từ list a
                    file_da_sua.append(b)
            with open(os.path.join(directory_path, filename), 'w') as file:
                for i in file_da_sua:
                    file.write(i + "\n")

elif format == "none":
    for i in os.listdir(src):
        if i.endswith('.jpg'):
            continue
        directory_path = osp.join(src, i)
        file_da_sua = []
        with open(directory_path, 'r') as file:
            for line in file:
                a = line.strip().split() # a: list phân tách cách thành phần của một dòng
                if a[0] == '0':
                    a[0] = '2'
                elif a[0] == '1':
                    a[0] = '0'
                elif a[0] == '4':
                    a[0] = '3'
                elif a[0] == '5':
                    a[0] = '3'
                elif a[0] == '6':
                    a[0] = '1'
                b = ' '.join(a) # b: yolo annotation string line, sau gộp từ list a
                file_da_sua.append(b)
        with open(directory_path, 'w') as file:
            for i in file_da_sua:
                file.write(i + "\n")