
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

SYSTEM_PROMPT = """You are BugHunter AI, an expert code reviewer.

Analyze the given code diff and respond in this exact format with no emojis:

Bug Risk: [one sentence on what could go wrong]
Risky Line: [quote the exact risky line from the diff]
Impact: [what happens if this reaches production]
Fix: [exact code fix suggestion]
Principle: [one coding principle being violated]

Rules:
- Be specific, reference exact lines and tokens
- Keep response under 150 words
- No emojis
- Simple language any developer can understand
- If code looks clean, still give one improvement tip"""

def get_fix_suggestion(diff, risky_tokens, prediction, confidence):
    user_prompt = f"""Prediction: {prediction} ({round(confidence, 2)}% confidence)
Risky tokens: {', '.join(risky_tokens)}

Code diff:
{diff[:300]}"""

    # Try Claude API first
    claude_key = os.environ.get("CLAUDE_API_KEY")
    if claude_key:
        print("Trying Claude API...")
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
        print(f"Claude status: {response.status_code}")
        if response.status_code == 200:
            return response.json()["content"][0]["text"]

    # Try Groq API
    groq_key = os.environ.get("GROQ_API_KEY")
    print(f"Groq key exists: {bool(groq_key)}")
    if groq_key:
        print("Trying Groq API...")
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
        print(f"Groq status: {response.status_code}")
        print(f"Groq response: {response.text[:200]}")
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

    # Fallback
    print("Both APIs failed, using fallback")
    return """Bug Risk: Potentially risky code patterns detected.
Risky Line: Check tokens flagged by SHAP analysis above.
Impact: Could cause unexpected behavior in production.
Fix: Review flagged tokens and add appropriate null checks and error handling.
Principle: Follow defensive programming principles."""

fix_suggestion = get_fix_suggestion(diff, top_buggy, prediction, confidence)

result = {
    "prediction": prediction,
    "confidence": round(confidence, 2),
    "top_buggy_tokens": [t.strip() for t in top_buggy if t.strip()],
    "top_clean_tokens": [t.strip() for t in top_clean if t.strip()],
    "fix_suggestion": fix_suggestion
}

with open("response.json", "w") as f:
    json.dump(result, f)
print(result)
