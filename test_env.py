import os
from pathlib import Path


def load_dotenv(path=None):
    if path is None:
        path = Path('.') / '.env'
    try:
        from dotenv import load_dotenv as _load_dotenv
        return _load_dotenv(path)
    except ImportError:
        if not path.exists():
            return False
        for line in path.read_text().splitlines():
            if not line or line.lstrip().startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = map(str.strip, line.split('=', 1))
            if value.startswith(('"', "'")) and value.endswith(("\"", "'")):
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return True

load_dotenv()

print("🔑 Groq Key tìm thấy:", os.getenv("GROQ_API_KEY"))
print("🔑 Gemini Keys tìm thấy:", os.getenv("GEMINI_API_KEYS"))