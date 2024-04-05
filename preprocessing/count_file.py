import os
import os.path as osp

folder_path = "../data4classes/"

yolo_format = ["train", "valid", "test"]

sum_file = 0

for i_yolo in yolo_format:
    path = osp.join(folder_path, i_yolo, "images")
    temp_count = len(os.listdir(path))
    sum_file += temp_count
    print(i_yolo + ": " + str(temp_count))

print("sum: " + str(sum_file))