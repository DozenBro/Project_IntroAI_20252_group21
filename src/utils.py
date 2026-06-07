import os
import yaml

def ensure_directories():
    """Tự động tạo các thư mục cần thiết nếu chúng chưa tồn tại."""
    directories = [
        'outputs/models',
        'outputs/figures',
        'outputs/reports'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
def load_config(config_path='config.yaml'):
    """Đọc file cấu hình YAML."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)