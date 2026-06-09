import os
from dotenv import load_dotenv
from groq import Groq
from google import genai

# Tự động nạp biến môi trường từ file .env
load_dotenv()

class AdvisoryLayer:
    """Manages the generation of personalized academic advice using Multi-LLM (Groq & Gemini)."""
    
    def __init__(self):
        self.meta_prompt = """
        🎯 SYSTEM ROLE: You are an expert Educational Data Analyst and Academic Advisor at a technical university. 
        Your objective is to interpret machine learning regression outputs (Predicted Exam Scores and LIME feature impacts) into professional, empathetic, and highly actionable natural language reports. 
        Maintain an analytical and objective tone. Focus on how specific behaviors directly added (+) or subtracted (-) points from the student's exam score. Do not use generic, robotic phrases.
        """
        
        self.format_constraints = """
        [OUTPUT REQUIREMENTS]
        Based on the SPECIFIC FACTORS above, please provide a concise analysis report:
        1. Score Diagnosis: Briefly explain why the student received this specific predicted score based on the extracted "Score Boosters" and "Score Reducers".
        2. Actionable Insights: Provide 2-3 targeted suggestions to recover the lost points. Ensure the advice strictly aligns with the factors provided. Keep the format clean and readable using Markdown.
        """

        # 1. Khởi tạo Groq Client
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None

        # 2. Khởi tạo Gemini Client (Sử dụng SDK google.genai MỚI NHẤT)
        self.gemini_api_key = os.getenv("GEMINI_API_KEYS")
        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.gemini_client = None

    def get_advice(self, lime_reasons, predicted_score, actual_score, use_mock=False, provider="Groq (Llama-3.3)"):
        """Generates advice via selected AI provider with Fallback mechanism."""
        prompt = self.generate_prompt(lime_reasons, predicted_score, actual_score)
        
        if use_mock:
            return prompt, self._generate_mock_advice(predicted_score, actual_score, lime_reasons)
            
        # Luồng xử lý định tuyến (Routing)
        if "Groq" in provider:
            return self._call_groq(prompt)
        elif "Gemini" in provider:
            return self._call_gemini(prompt)
        else:
            return prompt, "🚨 Không tìm thấy AI Provider."

    def _call_groq(self, prompt):
        """Gọi Groq, nếu lỗi tự động Fallback sang Gemini"""
        if not self.groq_client: return prompt, "🚨 Thiếu GROQ_API_KEY trong file .env"
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.meta_prompt},
                    {"role": "user", "content": prompt + "\n" + self.format_constraints}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5, 
            )
            return prompt, chat_completion.choices[0].message.content
        except Exception as e:
            error_msg = f"*(Hệ thống tự động Fallback do Groq lỗi: {e})*\n\n"
            _, gemini_response = self._call_gemini(prompt)
            return prompt, error_msg + gemini_response

    def _call_gemini(self, prompt):
        """Gọi Gemini, nếu lỗi tự động Fallback sang Groq"""
        if not self.gemini_client: return prompt, "🚨 Thiếu GEMINI_API_KEYS trong file .env"
        try:
            full_prompt = self.meta_prompt + "\n" + prompt + "\n" + self.format_constraints
            
            # Cú pháp gọi API mới toanh của Google
            response = self.gemini_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=full_prompt,
            )
            return prompt, response.text
        except Exception as e:
            error_msg = f"*(Hệ thống tự động Fallback do Gemini lỗi: {e})*\n\n"
            _, groq_response = self._call_groq(prompt)
            return prompt, error_msg + groq_response

    def generate_prompt(self, lime_reasons, predicted_score, actual_score):
        positive_factors = []
        negative_factors = []
        
        for condition, weight in lime_reasons:
            if abs(weight) < 0.5: continue 
            if weight > 0:
                positive_factors.append(f"🟢 Score Booster (Added Points): {condition} (Impact: +{weight:.2f} pts)")
            else:
                negative_factors.append(f"🔴 Score Reducer (Lost Points): {condition} (Impact: {weight:.2f} pts)")

        pos_text = "\n".join(positive_factors) if positive_factors else "- No significant score boosters detected."
        neg_text = "\n".join(negative_factors) if negative_factors else "- No significant score reducers detected."

        status = "NEEDS IMPROVEMENT" if predicted_score < 65 else "STABLE/GOOD"

        return f"""
        [STUDENT PERFORMANCE OVERVIEW]
        - AI Predicted Exam Score: {predicted_score:.1f} / 100 ({status})
        - Ground Truth Score: {actual_score:.1f} / 100
        
        [AI EXTRACTED ROOT CAUSES (from LIME XAI)]
        STRENGTHS (Factors that increased the score):
        {pos_text}
        
        BARRIERS (Factors that dragged the score down):
        {neg_text}
        """

    def _generate_mock_advice(self, predicted_score, actual_score, lime_reasons):
        advice = f"## 📊 Academic Diagnosis\n"
        advice += f"Based on the AI prediction, you are projected to score **{predicted_score:.1f}/100** (Actual: {actual_score:.1f}/100).\n\n"
        advice += "## 💡 Actionable Insights (Offline Mode)\n"
        
        reducers = [condition for condition, weight in lime_reasons if weight < -0.5]
        
        if reducers:
            advice += "The AI has identified the following areas dragging your score down:\n"
            for factor in reducers[:2]: 
                feature_name = factor.split(' <=')[0].split(' >')[0].replace('_', ' ')
                advice += f"* **Improve {feature_name}:** LIME analysis shows this factor is actively reducing your score.\n"
        else:
            advice += "* **Maintain Consistency:** No major negative factors were detected by LIME.\n"
        return advice