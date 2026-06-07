import os
import time
import random
from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()

class AdvisoryLayer:
    """Handles prompt engineering and interactions with LLM APIs (Groq/Gemini)."""
    
    def __init__(self):
        # Load external prompt configurations
        with open('prompts/meta_prompt.txt', 'r', encoding='utf-8') as f:
            self.meta_prompt = f.read()
        with open('prompts/format_constraints.txt', 'r', encoding='utf-8') as f:
            self.format_constraints = f.read()
            
        # Setup Gemini fallback mechanisms
        keys_str = os.getenv("GEMINI_API_KEYS", "")
        self.api_keys = [k.strip() for k in keys_str.split(',') if k.strip()]
        self.model_name = 'gemini-2.5-flash'
        
        # Setup primary Groq client
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None

    def _get_gemini_client(self):
        """Returns a Gemini client using a random API key for rate limit distribution."""
        if not self.api_keys: return None
        return genai.Client(api_key=random.choice(self.api_keys))

    def generate_prompt(self, lime_reasons, status_text, actual_status): 
        filtered_reasons = []
        for condition, weight in lime_reasons:
            # 1. Hạ ngưỡng lọc để LLM thấy được các chi tiết nhỏ
            if abs(weight) < 0.02: continue 
            
            # 2. [ĐÃ FIX LỖI] Cho LLM thấy cả Rào cản (weight > 0) và Lợi thế (weight < 0)
            if weight > 0:
                reason_desc = f"🔴 Barrier (Pushes towards Dropout): {condition} (Impact: {weight:.2f})"
            else:
                reason_desc = f"🟢 Advantage (Prevents Dropout): {condition} (Impact: {weight:.2f})"
                
            # 3. Gài luật để chống suy diễn bậy
            condition_lower = condition.lower()
            if "approved" in condition_lower:
                reason_desc += " -> NOTE: 'approved' means passed subjects."
            elif "age" in condition_lower:
                reason_desc += " -> NOTE: Standard enrollment age."
            elif "debtor" in condition_lower:
                if "<=" not in condition: 
                    reason_desc += " -> NOTE: This indicates debt status."
                    
            filtered_reasons.append("- " + reason_desc)

        reasons_text = "\n".join(filtered_reasons)
        
        # Nếu thực sự không có gì nổi bật (quá hiếm)
        if not reasons_text.strip():
            reasons_text = "- DATA NOTE: All metrics are very close to the average baseline. No extreme positive or negative outliers detected."

        # 4. [ĐÃ FIX LỖI] Thêm lệnh cấm chém gió "văn mẫu"
        if actual_status == "Graduate":
            role_instruction = """
            🎯 ROLE: SUCCESS CASE STUDY ANALYST.
            The student has GRADUATED.
            TASK: Analyze factors that contributed to their success.
            🚨 CRITICAL RULE: DO NOT invent generic reasons (like 'time management' or 'consistency'). Base your analysis STRICTLY on the 🟢 Advantages provided below.
            """
        elif actual_status == "Dropout":
            role_instruction = """
            🎯 ROLE: POST-MORTEM RISK ANALYST.
            The student DROPPED OUT.
            TASK: Deeply analyze the core 🔴 Barriers that caused the failure.
            """
        else:
            role_instruction = """
            🎯 ROLE: ACADEMIC ADVISOR.
            The student is CURRENTLY ENROLLED. 
            TASK: Based on predicted risk and current barriers, provide early warnings and actionable advice.
            """

        return f"""
        [STUDENT OVERVIEW]
        - AI Predicted Risk: {status_text}
        - Ground Truth Status: {actual_status}
        
        {role_instruction}
        
        [AI EXTRACTED FACTORS (LIME)]: 
        {reasons_text}
        
        {self.format_constraints}
        """

    def get_advice(self, lime_reasons, status_text, actual_status, use_mock=False):
        """Orchestrates API calls with primary (Groq) and fallback (Gemini) logic."""
        user_prompt = self.generate_prompt(lime_reasons, status_text, actual_status)
        full_prompt = f"{self.meta_prompt}\n\n{user_prompt}"
        
        # Attempt Primary API (Groq)
        if self.groq_client:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": full_prompt}],
                    model="llama-3.3-70b-versatile", 
                )
                return user_prompt, chat_completion.choices[0].message.content
            except Exception as e:
                print(f"Groq API error, falling back to Gemini: {e}")

        # Attempt Fallback API (Gemini) with retry mechanism
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
                    print(f"Gemini API error (attempt {attempt+1}): {e}")
                    time.sleep(1)
        
        return user_prompt, "⚠️ AI services are currently unavailable or quota exceeded."