
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import shap
import pandas as pd

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

result = {
    "prediction": prediction,
    "confidence": round(confidence, 2),
    "top_buggy_tokens": top_buggy,
    "top_clean_tokens": top_clean
}

with open("response.json", "w") as f:
    json.dump(result, f)
print(result)
