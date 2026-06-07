import os
import time
import random
from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()

class AdvisoryLayer:
    def __init__(self):
        # 1. Đọc prompts
        with open('prompts/meta_prompt.txt', 'r', encoding='utf-8') as f:
            self.meta_prompt = f.read()
        with open('prompts/format_constraints.txt', 'r', encoding='utf-8') as f:
            self.format_constraints = f.read()
            
        # 2. Cấu hình Gemini Key (Danh sách xoay vòng)
        keys_str = os.getenv("GEMINI_API_KEYS", "")
        self.api_keys = [k.strip() for k in keys_str.split(',') if k.strip()]
        self.model_name = 'gemini-2.5-flash'
        
        # 3. Cấu hình Groq
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None

    def _get_gemini_client(self):
        """Hàm lấy client với một API key ngẫu nhiên từ danh sách."""
        if not self.api_keys: return None
        return genai.Client(api_key=random.choice(self.api_keys))

    def generate_prompt(self, lime_reasons, status_text, actual_status): 
        filtered_reasons = []
        for condition, weight in lime_reasons:
            if abs(weight) < 0.05: continue 
            if weight > 0:
                reason_desc = f"Rào cản/Yếu tố chú ý: {condition} (Mức ảnh hưởng: {weight:.2f})"
                condition_lower = condition.lower()
                if "approved" in condition_lower:
                    reason_desc += " -> LƯU Ý: 'approved' là số môn THI ĐỖ. KHÔNG suy diễn là 'đỗ nhiều gây quá tải'."
                elif "age" in condition_lower:
                    reason_desc += " -> LƯU Ý: Đây là tuổi nhập học chuẩn (18 tuổi). KHÔNG nhận xét là 'thiếu trưởng thành'."
                elif "debtor" in condition_lower:
                    if "<=" in condition: continue 
                    else: reason_desc += " -> LƯU Ý: Đây là tình trạng nợ phí. Hãy phân tích khách quan."
                filtered_reasons.append("- " + reason_desc)

        reasons_text = "\n".join(filtered_reasons)
        if not reasons_text.strip():
            reasons_text = "- LƯU Ý: Không có yếu tố rào cản nào đáng kể. Sinh viên đang có các chỉ số rất ổn định."

        return f"""
        [THÔNG TIN SINH VIÊN]
        - Đánh giá của Hệ thống Cảnh báo (AI): {status_text}
        - Trạng thái thực tế: {actual_status} (Lưu ý: 'Enrolled' nghĩa là đang theo học bình thường, 'Graduate' là đã tốt nghiệp, 'Dropout' là đã bỏ học).
        
        [YÊU CẦU CHO BẠN (CỐ VẤN HỌC TẬP)]
        1. ĐỪNG CỐ TÌM KIẾM SỰ MÂU THUẪN. Việc AI đánh giá "Phong độ xuất sắc" cho một sinh viên "Enrolled" hay "Graduate" là sự ĐỒNG THUẬN tích cực.
        2. NHIỆM VỤ CHÍNH: Dựa vào [Đánh giá của Hệ thống] và các yếu tố bên dưới, hãy viết một đoạn nhận xét/tư vấn tự nhiên, ngắn gọn. 
        3. Phân tích cụ thể xem các yếu tố đó đang đem lại lợi thế gì, hoặc cảnh báo những rủi ro nhỏ nào có thể ảnh hưởng đến kết quả học tập. Đừng dùng từ ngữ máy móc.

        [Các yếu tố phân tích từ hệ thống LIME]: 
        {reasons_text}
        
        {self.format_constraints}
        """
    def get_advice(self, lime_reasons, status_text, actual_status, use_mock=False):
        user_prompt = self.generate_prompt(lime_reasons, status_text, actual_status)
        full_prompt = f"{self.meta_prompt}\n\n{user_prompt}"
        
        # 1. Thử gọi Groq trước
        if self.groq_client:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": full_prompt}],
                    # ĐÃ SỬA: Cập nhật sang model Llama mới nhất đang được Groq hỗ trợ
                    model="llama-3.3-70b-versatile", 
                )
                return user_prompt, chat_completion.choices[0].message.content
            except Exception as e:
                print(f"Groq lỗi, chuyển sang Gemini: {e}")

        # 2. Fallback sang Gemini với cơ chế xoay vòng Key
        if self.api_keys:
            max_retries = len(self.api_keys) * 2
            for attempt in range(max_retries):
                try:
                    client = self._get_gemini_client()
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=full_prompt
                    )
                    return user_prompt, response.text
                except Exception as e:
                    print(f"Lỗi Gemini (lần {attempt+1}): {e}")
                    time.sleep(1)
        
        return user_prompt, "⚠️ Hiện tại tất cả dịch vụ AI đều đang quá tải hoặc hết quota."