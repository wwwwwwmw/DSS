# Hướng dẫn chạy dự án DSS từ đầu đến cuối

## 0) Yêu cầu

- Windows
- Python 3.10+ (khuyến nghị)
- (Nếu dùng SQL Server) SQL Server + **ODBC Driver 18 for SQL Server**

## 1) Mở thư mục dự án

Mở folder `DSS/` trong VS Code.

## 2) Tạo môi trường và cài thư viện

Trong PowerShell tại `DSS/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> Vì sao bạn bị lỗi `running scripts is disabled`?
>
> - Windows PowerShell đang chặn chạy file script `.ps1` theo **ExecutionPolicy**.
> - Dự án **không bắt buộc** phải chạy `Activate.ps1`. Bạn có thể dùng trực tiếp `\.venv\Scripts\python.exe` như hướng dẫn ở trên.
>
> Nếu bạn vẫn muốn dùng lệnh activate, chạy 1 lần (khuyến nghị mức CurrentUser):
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

## 3) Cấu hình biến môi trường (.env)

Copy file mẫu:

```powershell
Copy-Item .env.example .env
```

Mở `.env` và chỉnh các biến sau:

- `SECRET_KEY`: chuỗi bất kỳ
- `DATABASE_URL`: chọn 1 trong 2

### 3.1) Dùng SQL Server (khuyến nghị)

Ví dụ (Windows Integrated Security):

```ini
DATABASE_URL=mssql+pyodbc://@DESKTOP-ENDTJTR/DSS_CarAdvisor?driver=ODBC+Driver+18+for+SQL+Server&Trusted_Connection=yes&TrustServerCertificate=yes
```

Nếu máy bạn dùng instance dạng `DESKTOP-ENDTJTR\SQLEXPRESS` thì thay `DESKTOP-ENDTJTR` thành `DESKTOP-ENDTJTR\\SQLEXPRESS`.

Gợi ý tạo database (tùy môi trường của bạn):

```sql
CREATE DATABASE DSS_CarAdvisor;
```

### 3.2) Dùng SQLite (chạy nhanh để demo)

```ini
DATABASE_URL=sqlite:///dss.sqlite3
```

## 4) Huấn luyện AI (train)

Chạy:

```powershell
.\.venv\Scripts\python.exe train.py --cars .\cars.csv --out .\models\car_advisor_rf.pkl
```

Khi train, terminal sẽ hiển thị **thanh % hoàn thành** và **thời gian chạy (elapsed)** để bạn biết chương trình vẫn hoạt động.
Ngoài ra sẽ có log file trong thư mục `logs/` (vd: `logs/train_YYYYmmdd_HHMMSS.log`).

### Train nhanh (khuyến nghị khi dataset lớn)

Chạy chế độ demo nhanh:

```powershell
.\.venv\Scripts\python.exe train.py --fast
```
.\.venv\Scripts\python.exe train.py --cars .\cars.csv --out .\models\car_advisor_rf.pkl
Hoặc lấy mẫu để train nhanh hơn (ví dụ 200k dòng):

```powershell
.\.venv\Scripts\python.exe train.py --sample-rows 200000
```

`--sample-rows` sẽ **không đọc toàn bộ file CSV**, nên phù hợp khi bạn thấy train "chạy rất lâu" do file lớn.

Hoặc lấy theo tỉ lệ (ví dụ 20%):

```powershell
.\.venv\Scripts\python.exe train.py --sample-frac 0.2
```

Kết quả sẽ tạo file model ở `models/`.

> Lưu ý: chi phí bảo dưỡng hiện đang là nhãn tổng hợp (synthetic) để demo.

## 5) Chạy Web

```powershell
.\.venv\Scripts\python.exe app.py
```

Mở trình duyệt: `http://127.0.0.1:5002`

## 6) Sử dụng hệ thống

### 6.1) Khách vãng lai

- Vào trang **Tư vấn**
- Nhập tối thiểu **3 xe**
- Chấm trọng số 1–9 cho các tiêu chí
- Xem kết quả theo 3 phương án:
  - **Xanh**: NÊN MUA NGAY
  - **Vàng**: CẦN CÂN NHẮC
  - **Đỏ**: RỦI RO CAO

### 6.2) So sánh xe

- Vào trang **So sánh**
- Nhập tối thiểu **2 xe**
- Hệ thống sẽ tô nổi bật ô có chỉ số tốt nhất theo từng tiêu chí

### 6.3) Người dùng đăng nhập

- Đăng ký/đăng nhập
- Sau khi tư vấn, hệ thống sẽ lưu **Lịch sử tư vấn**

### 6.4) Admin

Đăng nhập bằng tài khoản mặc định (tạo tự động lần đầu):

- `admin@example.com` / `admin123`

Tại trang **Admin** bạn có thể:

- Set role admin cho user
- Chỉnh **trọng số mặc định** (1–9)
- Upload CSV và retrain

## 7) Triển khai (tuỳ chọn)

Khuyến nghị dùng Waitress trên Windows:

```powershell
waitress-serve --listen=0.0.0.0:5002 --call app:create_app
```

## 8) Troubleshooting nhanh

- Nếu lỗi kết nối SQL Server: kiểm tra cài ODBC Driver, tên driver trong connection string, và quyền truy cập DB.
- Nếu file `models/car_advisor_rf.pkl` chưa có: chạy lại bước (4) hoặc dùng Admin → Retrain.

## 9) Chạy bằng 1 lệnh (tuỳ chọn)

Bạn có thể dùng các file `.bat` (không bị ExecutionPolicy chặn):

- Train: `train_model.bat`
- Chạy web: `run_app.bat`
