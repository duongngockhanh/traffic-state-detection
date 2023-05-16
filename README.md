# Traffic State Detection

## Object Detection

### Dataset
- It consists of 3,849 photos collected from various sources, from the Internet or from videos taken on highways and roads in Hanoi.
- Various data enhancement techniques from Roboflow API and available augmentation methods of YOLO were used.
- It includes 4 types of vehicles: car, bike, bus, truck. With the number of instances for each class as follows: 9,087 cars, 6,278 bikes, 1,138 buses, 3,976 trucks.
- The dataset is divided according to the ratio: 70/20/10.
- All images are processed in the standard size of 640x640 before being used for training.

### Configuration
- Three models were trained: YOLOv5n, YOLOv7, and YOLOv8n, using pre-trained models available in open-source, with a training of 100 epochs on the same dataset.
- YOLOv5n has a batch size of 32, while YOLOv7 and YOLOv8n have a batch size of 16.
- They were trained on Google Colab with a GPU configuration of Tesla K80 and 12GB of RAM.
- They were tested on a MacBook Pro 2015 with an Intel Core i5 Dual Core CPU and 16GB of RAM.

### Result
| Model | params (M) | FLOPs (B) | Size | Speed (ms) | mAP@50 | mAP@50-95
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv5n | 1.9 | 4.5 | 640 | 104.6 | 96.2% | 67.5% |
| YOLOv7 | 36.9 | 104.7 | 640 | 1441.2 | 97.2% | 70.3% |
| YOLOv8n | 3.2 | 8.7 | 640 | 178.3 | 96.8% | 70.9% |

## Congestion Classification

### Algorithm
- In this problem, traffic congestion classification is determined through the calculation of the ratio x between the total area of bounding boxes surrounding the detected vehicles (V) and the area of the region of interest (R), assuming that each pixel is a unit of area.
- The region of interest is the area where the detection and counting of vehicles will be performed. It is drawn immediately after running the program.
- Below is the area of the region of interest, R.
<p align="center">
<img width="376" alt="image" src="https://github.com/duongngockhanh/traffic-state-detection/assets/87640587/8a855f2f-aacf-4109-b666-7e6e7ab14443">
</p>
- Below is the total area of the bounding boxes, V.
<p align="center">
<img width="378" alt="image" src="https://github.com/duongngockhanh/traffic-state-detection/assets/87640587/58cc04bb-e194-40e4-b568-17ab6f72f5d7">
</p>
The ratio V/R is then mapped to a balanced scale and passed through a sigmoid function to distinguish more clearly between two states: congestion and normal.

### Result
The result indicates a normal traffic state.
<p align="center">
<img width="431" alt="image" src="https://github.com/duongngockhanh/traffic-state-detection/assets/87640587/f07818bb-9d14-4f8c-985d-7f53da2c4531">
</p>
The result indicates a traffic congestion state.
<p align="center">
<img width="431" alt="image" src="https://github.com/duongngockhanh/traffic-state-detection/assets/87640587/e2f932d4-2b01-4c9b-88d7-7d7052e8de86">
 </p>

## Installation

### Installation
```commandline
pip install https://github.com/duongngockhanh/traffic-state-detection.git
cd traffic-state-detection
pip install -r requirements.txt
```

### Download the test data
- The test data can be downloaded here: https://drive.google.com/drive/folders/14DErTVKrIr2MrVu8YtDE33cts0kTt6Bu
- Then place the video file in the folder named **a1_video_test**.

### Run
```commandline
python detect.py --weights a2_weights/yolov5n_bigger_dataset.pt --source a1_video_test/video2.mp4 --view-img
```
After that, proceed to mark the 4 points on the video to define the region of interest and observe the results.
