### Ý tưởng hiện tại:
- Ý tưởng 1: Dựa trên số lượng
+ Sử dụng Object Detection để phát hiện các loại vật thể bằng YOLOv5.
+ Sau đó cho qua một Machine Learning model để đưa ra prediction tắc đường hay không. 
+ Cụ thể, với số lượng của mỗi class sẽ là một feature, sẽ cần phải label 1 lần nữa cho phần ML.
+ Nhược điểm: Chỉ dùng trên một tuyến đường cụ thể.
- Ý tưởng 2: Dựa trên tốc độ
+ Vẫn sử dụng Object Detection để phát hiện các loại vật thể bằng YOLOv5.
+ Sử dụng Object Tracking để tính tốc độ trung bình của các xe.
+ Định nghĩa rằng, tốc độ trung bình của dòng phương tiện dưới 5km/h, nhỏ hơn tốc độ của người đi bộ, thì được coi là tắc đường.
+ Threashold = 5
+ Khắc phục được nhược điểm của ý tưởng 1.
