#This file contains the code required to run the Bert model
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import numpy as np

class Fred_Runner:
    """ This is the Fred Runner class. Given a model it contains all the necessary code to run it"""
    def __init__(self,model_path):
        self.tokenizer, self.loaded_model = self.load_model_and_tokenizer(4, model_path)

    def load_model_and_tokenizer(labels=4, model_path='/Users/adi/Desktop/RandomCode/Fred/bert_model.pth'):
        # Gets the tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load the model from the file
        loaded_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=labels)
        loaded_model.load_state_dict(torch.load(model_path))
        loaded_model.eval()  # Set the model to evaluation mode

        return tokenizer, loaded_model

    def predict_with_loaded_model(self, input_text, tokenizer = None, loaded_model = None):
        if loaded_model is None:
            loaded_model = self.model_location
        if tokenizer is None:
            tokenizer = self.tokenizer
        #Tokenize and encode the text
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

        #Forward pass through the model
        with torch.no_grad():
            outputs = loaded_model(**inputs)

        return outputs

    def predict_column(self, df, text_column, category_mapping = {0: 'OT', 1: "NC", 2: "LM", 3: "TM"}):
        # Load model and tokenizer
        df[text_column] = df[text_column].astype(str)
        # Apply prediction function to each value in the specified column
        predictions = df[text_column].apply(lambda x: self.predict_with_loaded_model(x, self.tokenizer, self.loaded_model))

        # Extract predicted classes and probabilities
        df['Predicted_Class: ' + text_column] = predictions.apply(
            lambda x: torch.argmax(softmax(x.logits, dim=1), dim=1).item()).map(category_mapping)
        df['Probabilities: ' + text_column] = predictions.apply(lambda x: softmax(x.logits, dim=1).tolist())

        # Calculate confidence intervals
        df['Confidence_Interval: ' + text_column] = predictions.apply(
            lambda x: np.percentile(softmax(x.logits, dim=1).numpy(), [2.5, 97.5], axis=1))

        return df
    def predict_returnExcel (self,location_path,text_column, edit_orignal = True):
        df = pd.read_excel(location_path)
        df = self.predict_column(df,text_column,location_path)



    def predict_returnCSV(self, location):
        df = self.predict_column()
