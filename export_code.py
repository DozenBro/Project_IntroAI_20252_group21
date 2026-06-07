import os
import json

# Đã thêm thư mục notebooks vào danh sách quét
folders_to_scan = ['app', 'src', 'prompts', 'notebooks']
files_to_scan = ['main.py', 'config.yaml', 'requirements.txt', 'README.md'] 

with open('toan_bo_project.txt', 'w', encoding='utf-8') as outfile:
    # 1. Quét các file lẻ bên ngoài
    for file in files_to_scan:
        if os.path.exists(file):
            outfile.write(f"\n{'='*50}\nFILE: {file}\n{'='*50}\n")
            with open(file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                
    # 2. Quét trong các thư mục
    for folder in folders_to_scan:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                # Bỏ qua các file rác của Python và Jupyter
                if '__pycache__' in root or '.ipynb_checkpoints' in root:
                    continue
                
                for file in files:
                    filepath = os.path.join(root, file)
                    
                    # --- XỬ LÝ ĐẶC BIỆT CHO JUPYTER NOTEBOOK (.ipynb) ---
                    if file.endswith('.ipynb'):
                        outfile.write(f"\n{'='*50}\nFILE: {filepath} (Đã lọc sạch code)\n{'='*50}\n")
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                notebook = json.load(f)
                                for i, cell in enumerate(notebook.get('cells', [])):
                                    cell_type = cell.get('cell_type', '')
                                    source = "".join(cell.get('source', []))
                                    if source.strip(): # Chỉ in ra nếu cell có nội dung
                                        outfile.write(f"\n# --- Cell {i+1} ({cell_type}) ---\n")
                                        outfile.write(source + "\n")
                        except Exception as e:
                            outfile.write(f"Lỗi khi đọc file notebook: {e}\n")
                            
                    # --- XỬ LÝ CÁC FILE THÔNG THƯỜNG ---
                    elif file.endswith(('.py', '.yaml', '.txt', '.md')):
                        outfile.write(f"\n{'='*50}\nFILE: {filepath}\n{'='*50}\n")
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())

print("🎉 Đã gộp thành công! (Bao gồm cả code đã được lọc sạch từ Jupyter Notebooks)")