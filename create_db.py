import os
import pandas as pd
import sqlite3

def initialize_database():
    print("--- ĐANG KHỞI TẠO CƠ SỞ DỮ LIỆU ---")
    csv_path = "data/raw/StudentPerformanceFactors.csv"
    db_dir = "data/database"
    db_path = f"{db_dir}/smartedu.db"

    if not os.path.exists(csv_path):
        print(f"[LỖI] Không tìm thấy file gốc tại {csv_path}. Vui lòng kiểm tra lại.")
        return
    
    os.makedirs(db_dir, exist_ok=True)

    #Đọc dữ liệu từ CSV và nạp vào SQLite
    try:
        df = pd.read_csv(csv_path)
        conn = sqlite3.connect(db_path)
        
        # Tạo bảng 'students_performance'
        df.to_sql('students_performance', conn, if_exists='replace', index=False)
        conn.close()
        print(f"[THÀNH CÔNG] Đã tạo Database SQLite tại: {db_path}")
        print(f"Tổng số bản ghi đã nạp: {len(df)}")
    except Exception as e:
        print(f"[LỖI] Quá trình tạo DB thất bại: {e}")

if __name__ == "__main__":
    initialize_database()