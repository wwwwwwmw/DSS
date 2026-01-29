# DSS – Hệ thống hỗ trợ mua xe ô tô (AHP + Random Forest)

Dự án này là một **Decision Support System (DSS)** hỗ trợ người dùng đánh giá xe ô tô dựa trên:

- **AHP (phiên bản đơn giản)**: người dùng chấm mức độ quan trọng 1–9 cho các tiêu chí.
- **AI dự đoán**:
  - **Random Forest Classifier** dự đoán *nguy cơ tai nạn/hư hại* (dựa trên cột `accidents_or_damage`).
  - **Random Forest Regressor** dự đoán *chi phí bảo dưỡng dự kiến theo tháng/năm*.

Giao diện web chia 3 nhóm:

- **Khách vãng lai**: nhập xe, chấm tiêu chí, xem tư vấn và so sánh.
- **Người dùng đăng nhập**: có thêm **lưu lịch sử tư vấn**.
- **Quản trị viên (admin)**: quản lý user, chỉnh **trọng số mặc định** của tiêu chí, upload CSV và **retrain AI**.

## Dữ liệu

File dữ liệu mặc định: `cars.csv` (20 cột), gồm các trường chính như:

`price`, `mileage`, `year`, `accidents_or_damage`, `one_owner`, `driver_rating`, `seller_rating`, `mpg`, `price_drop`, …

> Lưu ý quan trọng: dataset hiện tại **không có nhãn chi phí bảo dưỡng**. Vì vậy trong `train.py` dự án tạo **nhãn chi phí bảo dưỡng tổng hợp (synthetic)** để demo pipeline ML. Khi bạn có dữ liệu thực (maintenance cost), hãy thay `make_synthetic_maintenance()` bằng nhãn thật.

## Mô hình & output

- Model được lưu tại: `models/car_advisor_rf.pkl`
- Gói model gồm: `preprocessor` + `accident_clf` + `maint_reg` + `meta`

## Tài khoản admin mặc định

Khi chạy lần đầu, hệ thống tự tạo admin:

- Email: `admin@example.com`
- Password: `admin123`

Bạn nên đổi ngay sau khi chạy.

## Công nghệ

- Flask, Jinja2
- scikit-learn (RandomForestClassifier/Regressor)
- SQLAlchemy + pyodbc (SQL Server) *(có fallback SQLite cho dev)*

## Tài liệu chạy dự án

Xem hướng dẫn chi tiết ở `huongdan.md`.
