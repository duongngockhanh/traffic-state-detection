# Nghiên cứu mô hình học sâu phát hiện tình trạng giao thông

## Phát hiện đối tượng

### Tập dữ liệu
- Gồm 3.849 ảnh được sưu tầm từ nhiều nguồn khác nhau, từ Internet hoặc từ các video được quay trên các cao tốc và các tuyến đường tại Hà Nội.
- Được sử dụng các kỹ thuật tăng cường dữ liệu khác nhau của Roboflow API và các phương pháp augumentation sẵn có của YOLO.
- Bao gồm 4 loại phương tiện: car, bike, bus, truck. Với số lượng instance của mỗi class như sau: 9087 cars, 6278 bikes, 1138 buses, 3976 trucks.
- Tập dữ liệu được phân chia theo tỷ lệ: 70/20/10
- Tất cả các ảnh đều được xử lý ở kích thước chuẩn 640x640 trước khi dùng để huấn luyện.

### Cấu hình
- Ba mô hình được đào tạo: YOLOv5n, YOLOv7 và YOLOv8n, sử dụng các mô hình được đào tạo trước có sẵn trên mã nguồn mở, với số lượng 100 epoch được đào tạo trên cùng một bộ dữ liệu.
- YOLOv5n có batch size là 32, trong khi YOLOv7 và YOLOv8n có batch size là 16.
- Được đào tạo trên Google Colab với cấu hình GPU Tesla K80 và RAM 12GB.
- Đã thử nghiệm trên MacBook Pro 2015 với CPU Intel Core i5 Dual Core và RAM 16GB.

### Kết quả
| Model | params (M) | FLOPs (B) | Size | Speed (ms) | mAP@50 | mAP@50-95
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv5n | 1.9 | 4.5 | 640 | 104.6 | 96.2% | 67.5% |
| YOLOv7 | 36.9 | 104.7 | 640 | 1441.2 | 97.2% | 70.3% |
| YOLOv8n | 3.2 | 8.7 | 640 | 178.3 | 96.8% | 70.9% |

## Phân loại tắc đường

### Mô tả thuật toán
- Trong bài toán này, việc phân loại tình trạng giao thông được xác định thông qua quá trình tính toán tỷ lệ x giữa tổng diện tích các bounding box bao quanh các phương tiện được phát hiện (V) và diện tích của vùng quan tâm (R), giả sử rằng mỗi pixel là một đơn vị diện tích. 
- Với vùng quan tâm là vùng mà ta sẽ thực hiện tác vụ phát hiện và đếm số lượng các phương tiện trên đó. Được vẽ ngay sau khi khởi chạy chương trình.
Dưới đây là diện tích vùng quan tâm R
<p align="center">
<img width="376" alt="image" src="https://github.com/duongngockhanh/traffic-state-detection/assets/87640587/8a855f2f-aacf-4109-b666-7e6e7ab14443">
</p>
Dưới đây là tổng diện tích các bounding boxes V
<p align="center">
<img width="378" alt="image" src="https://github.com/duongngockhanh/traffic-state-detection/assets/87640587/58cc04bb-e194-40e4-b568-17ab6f72f5d7">
</p>
Tỷ lệ V/R sau đó được ánh xạ cân đối và cho qua hàm sigmoid để phân định rõ ràng hơn giữa hai trạng thái: ùn tắc và bình thường

### Kết quả
Kết quả cho tình trạng bình thường
<p align="center">
<img width="431" alt="image" src="https://github.com/duongngockhanh/traffic-state-detection/assets/87640587/f07818bb-9d14-4f8c-985d-7f53da2c4531">
</p>
Kết quả cho tình trạng tắc đường
<p align="center">
<img width="431" alt="image" src="https://github.com/duongngockhanh/traffic-state-detection/assets/87640587/e2f932d4-2b01-4c9b-88d7-7d7052e8de86">
 </p>

## Hướng dẫn cài đặt

### Cài đặt
```commandline
pip install https://github.com/duongngockhanh/traffic-state-detection.git
cd traffic-state-detection
pip install -r requirements.txt
```

### Tải dữ liệu kiểm thử
- Dữ liệu kiểm thử được tải xuống ở đây: https://drive.google.com/drive/folders/14DErTVKrIr2MrVu8YtDE33cts0kTt6Bu
- Sau đó đặt tệp video vào thư mục a1_video_test

### Chạy chương trình
```commandline
python detect.py --weights a2_weights/yolov5n_bigger_dataset.pt --source a1_video_test/video2.mp4 --view-img
```
Sau đó, tiến hành chấm 4 điểm trên video để xác định vùng quan tâm và quan sát kết quả.
