
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import shap
import pandas as pd
import os
import requests

# Load model
tokenizer = AutoTokenizer.from_pretrained("eldho15/ai-bug-hunter")
model = AutoModelForSequenceClassification.from_pretrained("eldho15/ai-bug-hunter")
model.eval()

with open("diff.txt") as f:
    diff = f.read()[:500]

# Prediction
inputs = tokenizer(diff, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
with torch.no_grad():
    output = model(**inputs)
probs = F.softmax(output.logits, dim=1)
confidence = float(probs[0][1]) * 100
prediction = "BUGGY" if confidence > 50 else "CLEAN"

# SHAP explanation
def predict_proba(texts):
    inputs = tokenizer(list(texts), max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return F.softmax(outputs.logits, dim=1).numpy()

explainer = shap.Explainer(predict_proba, tokenizer)
shap_values = explainer([diff], max_evals=200)
tokens = shap_values.data[0]
values = shap_values.values[0, :, 1]

shap_df = pd.DataFrame({"token": tokens, "shap_value": values})
shap_df = shap_df.sort_values("shap_value", ascending=False)

top_buggy = shap_df[shap_df["shap_value"] > 0].head(5)["token"].tolist()
top_clean = shap_df[shap_df["shap_value"] < 0].head(5)["token"].tolist()

# BugHunter AI character
SYSTEM_PROMPT = """You are BugHunter AI — an elite code security and quality analyst with expertise in:
- Software defect detection and prevention
- Secure coding practices
- Code smell identification
- Software engineering best practices

Your personality:
- Direct and precise — no fluff
- Developer-friendly — explain in simple terms
- Constructive — always provide actionable fixes
- Educational — briefly explain WHY something is risky

Your response format is ALWAYS:
1. 🐛 Bug Risk: One sentence on what could go wrong
2. 📍 Risky Line: Quote the exact line from the diff that is most dangerous
3. ⚠️ Impact: What would happen if this bug reaches production
4. 🔧 Fix: Exact code suggestion to fix the issue
5. 📚 Learn More: One line on the coding principle violated

Rules:
- Never be vague — always reference specific tokens or lines
- If multiple risks exist, focus on the most critical one
- Keep total response under 200 words
- Use markdown formatting
- If the code looks clean, still mention 1 improvement suggestion"""

# LLM fix suggestion
def get_fix_suggestion(diff, risky_tokens, prediction):
    user_prompt = f"""Commit flagged as: {prediction}
Confidence: {round(confidence, 2)}%
Risky tokens detected by SHAP: {', '.join(risky_tokens)}

Code diff:
{diff[:300]}

Analyze this commit and respond in your exact format."""

    # Try Claude API first
    claude_key = os.environ.get("CLAUDE_API_KEY")
    if claude_key:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": claude_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 400,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}]
            }
        )
        if response.status_code == 200:
            return response.json()["content"][0]["text"]

    # Try Groq API
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 400
            }
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

    # Fallback — no API available
    return """🐛 Bug Risk: Potentially risky code patterns detected.
📍 Risky Line: Check tokens flagged by SHAP analysis above.
⚠️ Impact: Could cause unexpected behavior in production.
🔧 Fix: Review flagged tokens and add appropriate null checks and error handling.
📚 Learn More: Follow defensive programming principles."""

fix_suggestion = get_fix_suggestion(diff, top_buggy, prediction)

result = {
    "prediction": prediction,
    "confidence": round(confidence, 2),
    "top_buggy_tokens": top_buggy,
    "top_clean_tokens": top_clean,
    "fix_suggestion": fix_suggestion
}

with open("response.json", "w") as f:
    json.dump(result, f)
print(result)
