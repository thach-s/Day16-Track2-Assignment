# Báo cáo ngắn — AWS CPU LightGBM benchmark

Benchmark được chạy trên EC2 `c7i-flex.large` (2 vCPU, 4 GiB RAM), là loại máy Free Tier hợp lệ gần tương đương `t3.medium` theo yêu cầu bài lab.
Dữ liệu Credit Card Fraud Detection gồm 284.807 dòng và 30 đặc trưng; thời gian đọc dữ liệu là 0,9064 giây.
LightGBM dừng sớm tại iteration 58 và hoàn thành huấn luyện trong 3,1502 giây.
Mô hình đạt AUC-ROC 0,977386 và accuracy 0,995716, cho thấy khả năng xếp hạng và phân loại tổng thể tốt.
Recall đạt 0,897959, nghĩa là mô hình nhận diện được phần lớn giao dịch gian lận trong tập test.
Precision 0,273292 và F1-score 0,419048 còn thấp do dữ liệu rất mất cân bằng và ngưỡng phân loại đang giữ ở 0,5.
Median latency khi dự đoán một dòng là 0,5286 ms; throughput cho batch 1.000 dòng đạt khoảng 662.403 dòng/giây.
Tại thời điểm kiểm tra sau benchmark, máy có khoảng 3,2 GiB RAM available và không có lỗi mạng hay tiến trình zombie.
Kết quả cho thấy mô hình LightGBM phù hợp với inference CPU nhanh; precision có thể cải thiện bằng cách điều chỉnh threshold trên validation set.
AWS Billing chưa hiển thị chi tiết dịch vụ tại thời điểm chụp vì tài khoản mới cần tối đa 24 giờ để chuẩn bị dữ liệu cost and usage.
