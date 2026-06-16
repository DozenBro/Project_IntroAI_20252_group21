# Project_IntroAI_20252_group21

# Hệ thống Dự đoán và Chẩn đoán Kết quả Học tập Sinh viên ứng dụng Học máy và GenAI
Giới thiệu dự án
Dự án này (Nhóm 21 - Nhập môn Trí tuệ Nhân tạo) cung cấp một hệ thống phần mềm toàn diện nhằm dự đoán điểm số của sinh viên dựa trên các yếu tố hành vi và môi trường. Khác với các mô hình "hộp đen" truyền thống, hệ thống ứng dụng Trí tuệ nhân tạo có thể giải thích (XAI) để bóc tách nguyên nhân, và Trí tuệ nhân tạo tạo sinh (GenAI) để tự động sinh báo cáo tư vấn học tập cá nhân hóa.

# Các tính năng cốt lõi
Dự đoán hiệu suất: Ứng dụng mô hình Ensemble (Voting Regressor) kết hợp Random Forest, Ridge, XGBoost và CatBoost để đưa ra điểm dự đoán có độ chính xác cao.

Chẩn đoán nguyên nhân: Sử dụng thuật toán LIME để định lượng các yếu tố làm tăng điểm (Score Boosters) và giảm điểm (Score Reducers) cho từng sinh viên.

Tư vấn tự động: Dịch kết quả toán học từ LIME thành lời khuyên thực tiễn thông qua Mô hình ngôn ngữ lớn (LLM).

Kiến trúc mô-đun hóa: Mã nguồn được chia thành 4 tầng xử lý độc lập, cho phép dễ dàng bảo trì và mở rộng.

# Tech stack: 
Ngôn ngữ lập trình: Python

Học máy (Machine Learning): Scikit-learn, XGBoost, CatBoost

Giải thích mô hình (XAI): LIME

LLM & Prompt Engineering: Google GenAI, Groq (Llama-3.3)

Giao diện người dùng: Streamlit

# Cấu trúc thư mục

Plaintext
PROJECT_INTROAI_20252_GROUP21/
├── app/
│   └── app.py                      # Giao diện Web Dashboard bằng Streamlit
├── data/
│   └── raw/
│       ├── Student_Performance.csv         # Dataset 1 (Dữ liệu cơ bản)
│       └── StudentPerformanceFactors.csv   # Dataset 2 (Dữ liệu đa biến - được sử dụng chính)
├── notebooks/
│   └── Dropout_Prediction_Pipeline.ipynb   # File phân tích dữ liệu (EDA) và huấn luyện mô hình
├── outputs/                        # Thư mục lưu trữ kết quả xuất ra (biểu đồ, model weights...)
├── prompts/
│   ├── format_constraints.txt      # Chứa các ràng buộc định dạng ép buộc LLM trả về đúng cấu trúc
│   └── meta_prompt.txt             # Chứa System Prompt định danh vai trò của AI
├── src/
│   ├── advisory_layer.py           # Tầng Tư vấn: Xử lý Prompt và gọi API LLM (Google/Groq)
│   ├── data_layer.py               # Tầng Dữ liệu: Tiền xử lý, điền khuyết và mã hóa
│   ├── prediction_layer.py         # Tầng Dự đoán: Quản lý và chạy mô hình Ensemble
│   ├── xai_layer.py                # Tầng Giải thích: Cấu hình và chạy thuật toán LIME
│   └── utils.py                    # Chứa các hàm tiện ích dùng chung
├── .env                            # Tệp chứa các API Key ẩn (Cần tự tạo)
├── config.yaml                     # Tệp cấu hình các tham số hệ thống
├── main.py                         # Điểm neo chạy luồng xử lý backend chính
├── requirements.txt                # Danh sách thư viện môi trường
└── README.md                       # Tài liệu dự án


# Hướng dẫn cài đặt và sử dụng
1. Khởi tạo môi trường ảo và cài đặt thư viện
Mở terminal tại thư mục gốc của dự án và chạy các lệnh sau:

Bash
python -m venv venv
# Kích hoạt venv trên Windows:
venv\Scripts\activate
# Kích hoạt venv trên MacOS/Linux:
source venv/bin/activate

# Cài đặt thư viện

pip install -r requirements.txt

2. Cấu hình biến môi trường
Tạo một tệp tin tên là .env tại thư mục gốc (ngang hàng với main.py). Thêm các khóa API của bạn vào tệp này:

Đoạn mã:
GEMINI_API_KEY=your_google_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

3. Khởi chạy ứng dụng
Để mở giao diện người dùng, sử dụng lệnh gọi Streamlit trỏ vào tệp app.py trong thư mục app:

Bash:
streamlit run app/app.py

Giao diện chẩn đoán học tập sẽ tự động mở trên trình duyệt tại địa chỉ http://localhost:8501.