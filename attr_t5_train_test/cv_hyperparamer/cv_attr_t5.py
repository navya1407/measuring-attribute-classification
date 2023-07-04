#iimporting necesssary libraries
import os
os.environ["TZ"] = "Asia/Kolkata"
import json
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments,EarlyStoppingCallback
import torch
from transformers import Seq2SeqTrainer
from torch.utils.data import Dataset, DataLoader
from itertools import product
import optuna
import numpy as np
import random
import joblib
import optuna.visualization as vis
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
storage_name = 'sqlite:///example.db'
nlp = spacy.load("en_core_web_sm")
import re
import time 

import re
from sklearn.model_selection import KFold

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(20)

with open('/home/navya/gpu/data/train_40k.json') as jsonfile:
    train_data = json.load(jsonfile)

# Split the data into training, validation, and testing sets
#val_data = val_data[:50]
train_data = train_data[:20000]
print(len(train_data))
df_train = pd.DataFrame(train_data)

k = 3
kf = KFold(n_splits=3, shuffle=True, random_state=42)
def preprocess(row,tok):
    tok=tok
    input_str = row['input']
    target_str = row['output']
    tokenized_inputs = tok(input_str, max_length=256, truncation=True)
    tokenized_targets = tok(target_str, max_length=14, truncation=True)
    tokenized_inputs['labels'] = tokenized_targets['input_ids']
    return tokenized_inputs
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'attention_mask': self.data[idx]['attention_mask'],
            'labels': self.data[idx]['labels'],
          
        }


def post_process(text):
  text = text.lower()
  tokens = nltk.word_tokenize(text)
  for i in range(len(tokens)):
     w=tokens[i]
     w=w.replace(" ","")
     tokens[i]=w
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [word for word in tokens if word not in stop_words]
  preprocessed_text = ' '.join(tokens)
  doc = nlp(preprocessed_text)
  lemmatized_text = " ".join([token.lemma_ for token in doc])
  text = re.sub('[^a-zA-Z0-9\s]', '',  lemmatized_text)
  tokens = nltk.word_tokenize(text)
  return tokens
def compute_metrics(eval_pred):
  tokenizer = AutoTokenizer.from_pretrained("t5-base")
  preds, labels = eval_pred
  prec,recall=[],[]
  if isinstance(preds, tuple):
       preds = preds[0]
  decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   # Replace -100 in the labels as we can't decode them.
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  for i in range(len(decoded_preds)):
        pred_tokens = post_process(decoded_preds[i])
        label_tokens = post_process(decoded_labels[i])
        
         # calculate overlap
        overlap = len(set(pred_tokens).intersection(set(label_tokens)))
        if(len(pred_tokens)!=0):
           rp = overlap / len(pred_tokens)
        else:
           rp=0
        prec.append(rp)
        if(len(label_tokens)!=0):
    # calculate relaxed recall
           rr = overlap / len(label_tokens)
        else:
             rr=0
        recall.append(rr)
  res={'Relaxed Precision':np.mean(rp),'Relaxed Recall': np.mean(rr)}
      
  return res
  

   
   





def obj(trial: optuna.Trial):
  start_time = time.time()
  learning_rate=trial.suggest_loguniform("learning_rate", low=4e-5, high=0.01)
  weight_decay=trial.suggest_loguniform("weight_decay", 4e-5, 0.01)
  train_epochs=trial.suggest_int("num_train_epochs", low=10, high=50)
  dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
  
  print(f" Trial number : {trial.number} ----> lr_rate : {learning_rate} , weight_decay : {weight_decay} , epochs : { train_epochs} ,dropout_rate: {dropout_rate} " ) 
  
  
  
  tokenizer = AutoTokenizer.from_pretrained("t5-base")
  train_dataset = df_train.apply(lambda row: preprocess(row,tokenizer), axis=1)
 # val_dataset = df_val.apply(lambda row: preprocess(row,tokenizer), axis=1)
  train_dataset=train_dataset.to_list()
  #val_dataset=val_dataset.to_list()
  train_dataset = MyDataset(train_dataset)
  #val_dataset = MyDataset(val_dataset)
  dir='/home/navya/gpu/cv_attr/checkpoints/'+f"trial_{trial.number}"

  training_args = Seq2SeqTrainingArguments(
  
  output_dir=dir,
  evaluation_strategy="epoch",
  save_strategy="epoch",
   logging_strategy='epoch',
  learning_rate=learning_rate,
  per_device_train_batch_size=32,
  per_device_eval_batch_size=16,
  weight_decay=weight_decay,
  save_total_limit=0,
  num_train_epochs=train_epochs,
  predict_with_generate=True,
  load_best_model_at_end=True
    #metric_for_best_model='eval_loss',
    #greater_is_better=False
   )
  score=[]
  for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold+1}')
 
    train_split = torch.utils.data.Subset(train_dataset, train_idx)
    val_split = torch.utils.data.Subset(train_dataset, val_idx)
 
    model_name = "t5-base" 

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,dropout_rate=dropout_rate)
    model.to('cuda')
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=train_split,
    eval_dataset=val_split,
   
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    
) 

    trainer.train()
    
    # Compute evaluation metrics for the current fold
    eval_output = trainer.evaluate(val_split)
    print(eval_output)
    score.append(eval_output['eval_Relaxed Precision'])

  end_time = time.time()
  print("Time taken:", end_time - start_time, "seconds")
  print(f'score_list: {score} and score : {np.mean(score)}')
  return np.mean(score)

# We want to minimize the loss!
sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(study_name="hyper-parameters-cv",storage=storage_name, direction="maximize",sampler=sampler)
#study = optuna.load_study(study_name="hyper-parameters-sear", storage=storage_name)
study.optimize(func=obj, n_trials=10)

d={}
print("Best parameters")

for key, value in study.best_trial.params.items():
    d[key]=value
    print(f"    {key}: {value}")
with open('/home/navya/gpu/cv_attr/final_params.json', 'w') as f:
    json.dump(d, f)


joblib.dump(study, "/home/navya/gpu/cv_attr/study.pkl")
