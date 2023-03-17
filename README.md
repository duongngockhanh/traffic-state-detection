# Ý tưởng hiện tại:
## Ý tưởng 1: Dựa trên số lượng
+ Sử dụng Object Detection để phát hiện các loại vật thể bằng YOLOv5.
+ Sau đó cho qua một Machine Learning model để đưa ra prediction tắc đường hay không. 
+ Cụ thể, với số lượng của mỗi class sẽ là một feature, sẽ cần phải label 1 lần nữa cho phần ML.
+ Ưu điểm: Dễ triển khai.
+ Nhược điểm: Chỉ dùng trên một tuyến đường cụ thể.

## Ý tưởng 2: Dựa trên tốc độ
+ Vẫn sử dụng Object Detection để phát hiện các loại vật thể bằng YOLOv5.
+ Sử dụng Object Tracking để tính tốc độ trung bình của các xe.
+ Định nghĩa rằng, tốc độ trung bình của dòng phương tiện dưới 5km/h, nhỏ hơn tốc độ của người đi bộ, thì được coi là tắc đường.
+ Threashold = 5
+ Khắc phục được nhược điểm của ý tưởng 1.
### Ý tưởng 2.1: Dựa vào việc phát hiện lane để tính tốc độ
+ Huấn luyện mô hình phát hiện thêm cả lane, vẽ được nhiều region có độ dài bằng nhau bằng mỗi 2 lane.
+ Từ đó, có thể tính được vận tốc trên mỗi region.
+ Kết quả: Không thành công.

https://user-images.githubusercontent.com/87640587/225799612-3cee09ef-9c7d-4fc1-9513-e1f9c1116633.mp4
### Ý tưởng 2.2: Tạo thủ công 1 region để đo tốc độ trong region đó
+ Nhược điểm: Không khách quan, do xe phải đi qua region mới có thể phát hiện tốc độ.
### Ý tưởng 2.3: Sử dụng Bird Eye View Transformation
+ Chuyển tọa độ của đường, xe về dạng 2D. Sau đó tính vận tốc trung bình.
+ Ưu điểm: Khách quan, cho mọi tuyến đường.
+ Nhược điểm: Cần 2 camera. Độ khó cao.
