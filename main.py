import torch
import json
import os
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from tqdm import tqdm
import click


from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from edit_distance import Edit_distance



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
    text1: str
    text2: str

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def comparative_score(score1, score2, epsilon=1e-6):
    """
    Return a single score in [0, 1] based on the comparison of two [0, 1] input scores.

    :param score1: first score
    :param score2: second score
    :param epsilon: non-answer (output score = 0.5) epsilon threshold
    :return: [0, 0.5) if score1 > score2 + eps; (0.5, 1] if score2 > score1 + eps; 0.5 otherwise
    """
    if score1 > score2 + epsilon:
        return (1.0 - min(max(score2, 0.0), 1.0)) / 2.0 + 0.5
    if score2 > score1 + epsilon:
        return min(max(score1, 0.0), 1.0) / 2.0
    return 0.5





def CalculateScore(text1, text2):
    score1 = detector.get_score(text1)
    score2 = detector.get_score(text2)

    edit_distance1=1
    edit_distance2=1
    try:
        edit_distance1 = Edit_distance(text1)
        edit_distance2 = Edit_distance(text2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    score1= score1*0.75 + 0.25*(edit_distance1/(edit_distance1+edit_distance2))
    score2= score2*0.75 + 0.25*(edit_distance2/(edit_distance1+edit_distance2))
    score = comparative_score(score1, score2)
    return score
@app.get("/")
async def read_root():
    return {"message": "Welcome to the detectAI API"}

@app.post("/predict")
async def predict_score(text_request: TextRequest):
    try:
        score = CalculateScore(text_request.text1, text_request.text2) 
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

