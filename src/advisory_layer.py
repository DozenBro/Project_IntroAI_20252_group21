import os
from dotenv import load_dotenv
from groq import Groq
from google import genai

load_dotenv()

class AdvisoryLayer:
    """Manages the generation of personalized academic advice using Multi-LLM."""
    
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..")) 
        
        meta_path = os.path.join(project_root, "prompts", "meta_prompt.txt")
        format_path = os.path.join(project_root, "prompts", "format_constraints.txt")
        
        print("\n--- SYSTEM DIAGNOSTICS: PROMPT LOADING ---")
        print(f"Target Meta Path: {meta_path}")
        print(f"Target Format Path: {format_path}")
        
        # Đọc file với cơ chế Fallback
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta_prompt = f.read()
                
            with open(format_path, "r", encoding="utf-8") as f:
                self.format_constraints = f.read()
                
            print("STATUS: SUCCESS. Brain loaded from text files.")
        except FileNotFoundError as e:
            print(f"STATUS: FAILED. File not found. Error: {e}")
            self.meta_prompt = "SYSTEM ROLE: You are an Academic Advisor. OUTPUT CONSTRAINT: You MUST write your ENTIRE response in Vietnamese. No English allowed."
            self.format_constraints = "Format in Markdown. Go straight to Action Plan. DO NOT output English."
            
        print("\n")

        # API
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None

        gemini_api_key = os.getenv("GEMINI_API_KEYS")
        if gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key) 
        else:
            self.gemini_client = None
        
        # 
    def get_advice(self, lime_reasons, predicted_class, probabilities, actual_score, student_data, target_profile, use_mock=False, provider="Groq (Llama-3.3)"):
        prompt = self.generate_prompt(lime_reasons, predicted_class, probabilities, actual_score, student_data, target_profile)
        
        if use_mock:
            return prompt, self._generate_mock_advice(predicted_class, probabilities, actual_score, lime_reasons, student_data, target_profile)
            
        if "Groq" in provider:
            return self._call_groq(prompt)
        elif "Gemini" in provider:
            return self._call_gemini(prompt)
        else:
            return prompt, "Error: AI Provider not found."

    def _call_groq(self, prompt):
        if not self.groq_client: return prompt, "Error: Missing GROQ_API_KEY"
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
            error_msg = f"*(Fallback due to Groq error: {e})*\n\n"
            _, gemini_response = self._call_gemini(prompt)
            return prompt, error_msg + gemini_response

    def _call_gemini(self, prompt):
        if not self.gemini_client: 
            return prompt, "Error: Missing GEMINI_API_KEYS" 
        try:
            full_prompt = self.meta_prompt + "\n" + prompt + "\n" + self.format_constraints
            response = self.gemini_client.models.generate_content(
                model='gemini-2.0-flash',  
                contents=full_prompt,
            )
            return prompt, response.text
            
        except Exception as e:
            error_msg = f"*(Fallback due to Gemini error: {e})*\n\n"
            _, groq_response = self._call_groq(prompt)
            return prompt, error_msg + groq_response

    def generate_prompt(self, lime_reasons, predicted_class, probabilities, actual_score, student_data, target_profile):
        class_names = ['Needs Improvement', 'Average', 'Good', 'Excellent']
        current_tier = class_names[predicted_class]
        
        # Probabilities text for LLM context
        prob_text = ", ".join([f"{cls}: {p*100:.1f}%" for cls, p in zip(class_names, probabilities)])

        positive_factors = [f"+{w:.2f} impact: {c}" for c, w in lime_reasons if w > 0]
        negative_factors = [f"{w:.2f} impact: {c}" for c, w in lime_reasons if w < 0]

        pos_text = "\n".join(positive_factors) if positive_factors else "- None"
        neg_text = "\n".join(negative_factors) if negative_factors else "- None"

        # Tinh toan chinh xac khoang cach voi nhom target
        gap_analysis_text = ""
        key_features = ['Hours_Studied', 'Sleep_Hours', 'Attendance', 'Tutoring_Sessions']
        for feat in key_features:
            if feat in student_data.index and feat in target_profile.index:
                current_val = student_data[feat]
                target_val = target_profile[feat]
                diff = target_val - current_val
                gap_analysis_text += f"- {feat}: Current = {current_val:.1f} | Target Peer = {target_val:.1f} | Gap = {'+' if diff > 0 else ''}{diff:.1f}\n"

        # Giai ma Ordinal Encoding de cung cap context that cho LLM
        income_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        income_val = student_data.get('Family_Income', 1)
        income = income_mapping.get(int(income_val), 'Unknown')

        return f"""
        [STUDENT PERFORMANCE OVERVIEW]
        - Ground Truth Score: {actual_score:.1f} / 100
        - AI Predicted Academic Tier: {current_tier}
        - Tier Probabilities: {prob_text}
        
        [PERSONAL CONSTRAINTS]
        - Family Income Level: {income} (Strictly adhere to meta_prompt rule #3 based on this).
        
        [AI EXTRACTED ROOT CAUSES (from LIME XAI)]
        STRENGTHS (Factors maintaining current probability):
        {pos_text}
        
        BARRIERS (Factors pushing student to lower tiers):
        {neg_text}
        
        [GAP ANALYSIS VS GOOD/EXCELLENT PEERS]
        (To reach the target tier, the student needs to close these specific gaps):
        {gap_analysis_text}
        """

    def _generate_mock_advice(self, predicted_class, probabilities, actual_score, lime_reasons, student_data, target_profile):
        class_names = ['Needs Improvement', 'Average', 'Good', 'Excellent']
        advice = f"## Academic Diagnosis\n"
        advice += f"Predicted Tier: **{class_names[predicted_class]}** (Actual Score: {actual_score:.1f}/100).\n\n"
        advice += "## Actionable Insights (Offline Mode)\n"
        advice += "Targeting gaps:\n"
        
        key_features = ['Hours_Studied', 'Attendance']
        for feat in key_features:
            if feat in student_data.index:
                advice += f"- **{feat}**: Current {student_data[feat]:.1f} vs Target {target_profile[feat]:.1f}\n"
        
        return advice