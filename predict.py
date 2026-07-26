
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import shap
import pandas as pd
import os
import requests
import re

# Load model
tokenizer = AutoTokenizer.from_pretrained("eldho15/ai-bug-hunter")
model = AutoModelForSequenceClassification.from_pretrained("eldho15/ai-bug-hunter")
model.eval()

with open("diff.txt") as f:
    raw_diff = f.read()[:2000]

# Parse diff into file + line structure
def parse_diff(diff_text):
    results = []
    current_file = None
    line_number = 0

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            current_file = line.split(" b/")[-1]
        elif line.startswith("@@"):
            match = re.search(r"\+([0-9]+)", line)
            if match:
                line_number = int(match.group(1)) - 1
        elif line.startswith("+") and not line.startswith("+++"):
            line_number += 1
            results.append({
                "file": current_file,
                "line": line_number,
                "content": line[1:].strip()
            })
        elif not line.startswith("-"):
            line_number += 1

    return results

parsed_lines = parse_diff(raw_diff)
diff_text = " ".join([l["content"] for l in parsed_lines])[:500]

# Prediction
inputs = tokenizer(diff_text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
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
shap_values = explainer([diff_text], max_evals=200)
tokens = shap_values.data[0]
values = shap_values.values[0, :, 1]

shap_df = pd.DataFrame({"token": tokens, "shap_value": values})
shap_df = shap_df.sort_values("shap_value", ascending=False)
top_buggy = shap_df[shap_df["shap_value"] > 0].head(5)["token"].tolist()
top_clean = shap_df[shap_df["shap_value"] < 0].head(5)["token"].tolist()

# Match risky tokens to specific lines
def find_risky_lines(parsed_lines, risky_tokens):
    risky_lines = []
    for token in risky_tokens:
        token = token.strip()
        if not token or len(token) < 2:
            continue
        for entry in parsed_lines:
            if token.lower() in entry["content"].lower():
                risky_lines.append({
                    "file": entry["file"],
                    "line": entry["line"],
                    "content": entry["content"],
                    "token": token
                })
                break
    return risky_lines

risky_lines = find_risky_lines(parsed_lines, top_buggy)

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

def get_fix_suggestion(diff, risky_tokens, risky_lines, prediction, confidence):
    risky_lines_text = "\n".join([
        f"File: {r['file']}, Line {r['line']}: {r['content']}"
        for r in risky_lines
    ]) if risky_lines else "No specific lines identified"

    user_prompt = f"""Prediction: {prediction} ({round(confidence, 2)}% confidence)
Risky tokens: {', '.join(risky_tokens)}

Specific risky lines found:
{risky_lines_text}

Full diff:
{diff[:300]}"""

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

    return """Bug Risk: Potentially risky code patterns detected.
Risky Line: Check tokens flagged by SHAP analysis above.
Impact: Could cause unexpected behavior in production.
Fix: Review flagged tokens and add appropriate null checks and error handling.
Principle: Follow defensive programming principles."""

fix_suggestion = get_fix_suggestion(diff_text, top_buggy, risky_lines, prediction, confidence)

result = {
    "prediction": prediction,
    "confidence": round(confidence, 2),
    "top_buggy_tokens": [t.strip() for t in top_buggy if t.strip()],
    "top_clean_tokens": [t.strip() for t in top_clean if t.strip()],
    "risky_lines": risky_lines,
    "fix_suggestion": fix_suggestion
}

with open("response.json", "w") as f:
    json.dump(result, f)
print(result)
