import torch
import json
import os
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from tqdm import tqdm
import click


from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

# Assuming the model and tokenizer are loaded globally
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the model configuration and model from a .pth file
config = BertConfig.from_pretrained('bert-base-uncased')
BertModel = BertForSequenceClassification(config)
model = BertModel.from_pretrained("yadagiriannepaka/BERT_MODELGYANDEEP.pth")  #this is fetching from external directoory
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class TextDetector:
    def get_score(self, text):
        # Encode the text to BERT's format
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True,
                           padding=True)  # our tokenizing logic may be different
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)  # how to find this probability

        # Assuming we are interested in the probability of the first class
        return probabilities[0, 1].item()


detector = TextDetector()

class TextRequest(BaseModel):
    text: str

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the detectAI API"}

@app.post("/predict")
async def predict_score(text_request: TextRequest):
    try:
        score = detector.get_score(text_request.text)
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

