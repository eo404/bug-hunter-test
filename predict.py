
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

tokenizer = AutoTokenizer.from_pretrained("eldho15/ai-bug-hunter")
model = AutoModelForSequenceClassification.from_pretrained("eldho15/ai-bug-hunter")
model.eval()

with open("diff.txt") as f:
    diff = f.read()[:500]

inputs = tokenizer(diff, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
with torch.no_grad():
    output = model(**inputs)
probs = F.softmax(output.logits, dim=1)
confidence = float(probs[0][1]) * 100
prediction = "BUGGY" if confidence > 50 else "CLEAN"

result = {"prediction": prediction, "confidence": round(confidence, 2)}
with open("response.json", "w") as f:
    json.dump(result, f)
print(result)
