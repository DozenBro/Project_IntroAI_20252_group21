# Project Introduction
This project (Group 21 - Introduction to Artificial Intelligence) provides a comprehensive software system designed to classify students into academic performance tiers based on behavioral and environmental factors. Unlike traditional "black-box" machine learning models, the system applies Explainable AI (XAI) to isolate root causes, and Generative AI (GenAI) to automatically generate personalized academic advisory roadmaps based on real-world data analysis.

# Core Features
Academic Performance Classification: Applies an Ensemble model (Voting Classifier) combining the strengths of 4 algorithms: Random Forest, XGBoost, CatBoost, and Logistic Regression to classify students into 4 performance tiers (Needs Improvement, Average, Good, Excellent).

Root Cause Diagnosis (XAI): Utilizes the LIME algorithm to quantify and isolate the factors maintaining a student's current academic standing (Retention Factors) and the risk factors pulling their scores down (Risk Factors) for each specific individual.

Automated Advisory (GenAI): Translates mathematical weights from the LIME algorithm into a strategic, personalized action plan via Large Language Models (LLMs).

Layered Architecture: The source code is divided into 4 independent processing layers (Data Layer, Prediction Layer, XAI Layer, Advisory Layer), ensuring modularity, easy maintenance, and scalability.

# Tech Stack
Programming Language: Python

Machine Learning: Scikit-learn, XGBoost, CatBoost

Explainable AI (XAI): LIME

LLM & Prompt Engineering: Google GenAI (Gemini 2.0 Flash), Groq API (Llama-3.3-70b)

User Interface: Streamlit

Data Processing: Pandas, NumPy

# Cấu trúc thư mục

```text
PROJECT_INTROAI_20252_GROUP21/
├── create_db.py
├── data/
│   ├── raw/
│   │   ├── Student_Performance.csv         # Dataset 1 (Dữ liệu cơ bản)
│   │   └── StudentPerformanceFactors.csv   # Dataset 2 (Dữ liệu đa biến - chính)
├── notebooks/
│   └── Dropout_Prediction_Pipeline.ipynb   # File phân tích (EDA) và huấn luyện
├── outputs/                                # Thư mục lưu kết quả (biểu đồ, model...)
├── prompts/
│   ├── format_constraints.txt              # Ràng buộc định dạng cho LLM
│   └── meta_prompt.txt                     # System Prompt định danh vai trò AI
├── src/
│   ├── app/
│   │   └── app.py                          # Giao diện Web Dashboard (Streamlit)
│   ├── advisory_layer.py                   # Tầng Tư vấn: Gọi API LLM (Groq/Gemini)
│   ├── data_layer.py                       # Tầng Dữ liệu: Tiền xử lý, mã hóa
│   ├── prediction_layer.py                 # Tầng Dự đoán: Mô hình Ensemble
│   ├── xai_layer.py                        # Tầng Giải thích: Thuật toán LIME
│   └── utils.py                            # Các hàm tiện ích dùng chung
├── .env                                    # Tệp chứa API Key (Tự tạo)
├── config.yaml                             # Tệp cấu hình tham số hệ thống
├── main.py                                 # Chạy luồng xử lý backend tự động
├── requirements.txt                        # Danh sách thư viện cần cài đặt
└── README.md                               # Tài liệu hướng dẫn dự án
```

# Hướng dẫn cài đặt và sử dụng
1. Khởi tạo môi trường ảo và cài đặt thư viện
Mở terminal tại thư mục gốc của dự án và chạy các lệnh sau:

# MT ảo
python -m venv venv
# Kích hoạt venv trên Windows:
venv\Scripts\activate
# Kích hoạt venv trên MacOS/Linux:
source venv/bin/activate

# Cài đặt thư viện

pip install -r requirements.txt

# Cấu hình biến môi trường
Tạo một tệp tin tên là .env tại thư mục gốc. Thêm các khóa API vào tệp này:

Đoạn mã:
GEMINI_API_KEYS= "your_google_gemini_api_key_here"
GROQ_API_KEY= "your_groq_api_key_here"

# init database
python create_db.py

# Khởi chạy ứng dụng
Để mở giao diện người dùng, sử dụng lệnh gọi Streamlit trỏ vào tệp app.py trong thư mục app:

BA:
python -m src.main

Bash:
streamlit run src\app\app.py
