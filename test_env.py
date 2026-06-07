import os
from dotenv import load_dotenv

load_dotenv()

print("Groq Key tìm thấy:", os.getenv("GROQ_API_KEY"))
print("Gemini Keys tìm thấy:", os.getenv("GEMINI_API_KEYS"))