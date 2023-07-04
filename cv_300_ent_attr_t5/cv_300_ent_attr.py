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

with open('/home/navya/gpu/cv_attr/cv_f1_attr/t5_true/true_300_ent_attr.json') as jsonfile:
    train_data = json.load(jsonfile)

# Split the data into training, validation, and testing sets

print(len(train_data))
print(train_data[10])
df_train = pd.DataFrame(train_data)

k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)
def preprocess(row,max_input_len,max_target_len,tok):
    tok=tok
    input_str = row['input']
    target_str = row['output']
    tokenized_inputs = tok(input_str, max_length=max_input_len, truncation=True)
    tokenized_targets = tok(target_str, max_length=max_target_len, truncation=True)
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
#post processing for entity
def preprocess_ent(text):
  # lowercase the text
   text = text.lower()
   text=re.sub(r'\b(a|an|the)\b', '', text)
   text=text.replace("'s","")
   text=text.replace(" 's","")


# tokenize the text
   tokens = nltk.word_tokenize(text)
   for i in range(len(tokens)):
     w=tokens[i]

     w=w.replace(" ","")
     tokens[i]=w

   return tokens

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
  tokenizer.add_tokens(["<sep>"])
  preds, labels = eval_pred
  prec,recall,ent_p,ent_r=[],[],[],[]
  if isinstance(preds, tuple):
       preds = preds[0]
  decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   # Replace -100 in the labels as we can't decode them.
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
  for i in range(len(decoded_preds)):
        pred_text=decoded_preds[i]
        label_text=decoded_labels[i]
        #seperate entity and attribute for predicted text
        if('<sep>' in pred_text ):
          pred_ent=pred_text.split('<sep>')[0].strip()
          pred_attr=pred_text.split('<sep>')[1].strip()

        else:

          pred_ent= pred_text.strip()
          pred_attr=''
        #seperate entity and attribute for labelled text
        if('<sep>' in label_text ):
           label_ent=label_text.split('<sep>')[0].strip()
           label_attr=label_text.split('<sep>')[1].strip()
        else:
          label_ent= label_text.strip()
          label_attr=''
        #find the predicted and labelled entity
        pred_ent=preprocess_ent( pred_ent)
        label_ent=preprocess_ent( label_ent)
        #find the predicted and labelled attribute
        pred_tokens= post_process(pred_attr)
        label_tokens= post_process(label_attr)


   # for attribute

         # calculate overlap
        overlap = len(set(pred_tokens).intersection(set(label_tokens)))
        # calculate relaxed precision
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

    #for entity
        # calculate overlap
        overlap = len(set(pred_ent).intersection(set(label_ent)))
          # calculate relaxed precision
        if(len(pred_ent)!=0):
           rp = overlap / len(pred_ent)
        else:
           rp=0
        ent_p.append(rp)
        if(len(label_ent)!=0):
    # calculate relaxed recall
           rr = overlap / len(label_ent)
        else:
             rr=0
        ent_r.append(rr)


  precision,recall=np.mean(prec),np.mean(recall)
  f1=(2*precision*recall )/ (precision+recall)

  ent_precision,ent_recall=np.mean(ent_p),np.mean(ent_r)
  f1_ent= (2*ent_precision* ent_recall) / (ent_precision+ent_recall)




  res={'Attribute Relaxed Precision':precision,'Attribute Relaxed Recall':recall,'Attribute F1-score':f1 , 'Entity Relaxed Precision':ent_precision,'Entity Relaxed Recall':ent_recall,'Entity F1-score':f1_ent}

  return res



  

   
   
def obj(trial: optuna.Trial):
  start_time = time.time()
  learning_rate=trial.suggest_loguniform("learning_rate", low=4e-5, high=0.01)
  weight_decay=trial.suggest_loguniform("weight_decay", 4e-5, 0.01)
  train_epochs=trial.suggest_int("num_train_epochs", low=10, high=50)
  dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
  train_batch_size=trial.suggest_categorical("per_device_train_batch_size",[16,32,64,128])
  eval_batch_size=trial.suggest_categorical("per_device_eval_batch_size",[16,32,64,128])
  input_seq_length = trial.suggest_categorical('input_seq_length', [64, 128,256])
  output_seq_length = trial.suggest_categorical('output_seq_length', [6,10, 14])

  print(f" Trial number : {trial.number} ----> lr_rate : {learning_rate} , weight_decay : {weight_decay} , epochs : { train_epochs} ,train batch size: {train_batch_size} ,eval batch size: {eval_batch_size} , input_seq_length : {input_seq_length}, output_seq_length: {output_seq_length},dropout_rate: {dropout_rate} " ) 
  
  
  
  tokenizer = AutoTokenizer.from_pretrained("t5-base")
  train_dataset = df_train.apply(lambda row: preprocess(row,input_seq_length,output_seq_length,tokenizer), axis=1)
 # val_dataset = df_val.apply(lambda row: preprocess(row,tokenizer), axis=1)
  train_dataset=train_dataset.to_list()
  #val_dataset=val_dataset.to_list()
  train_dataset = MyDataset(train_dataset)
  #val_dataset = MyDataset(val_dataset)
  dir='/home/navya/gpu/cv_attr/cv_f1_attr/cv_300_ent_attr/checkpoints/'+f"trial_{trial.number}"

  training_args = Seq2SeqTrainingArguments(
  
  output_dir=dir,
  evaluation_strategy="epoch",
  save_strategy="epoch",
   logging_strategy='epoch',
  learning_rate=learning_rate,
  per_device_train_batch_size=train_batch_size,
  per_device_eval_batch_size=eval_batch_size,
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
    score.append(eval_output['Attribute F1-score'])

  end_time = time.time()
  print("Time taken:", end_time - start_time, "seconds")
  print(f'score_list: {score} and score : {np.mean(score)}')
  return np.mean(score)

# We want to minimize the loss!
sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(study_name="hyper-parameters-cv55555-ent_attr-300",storage=storage_name, direction="maximize",sampler=sampler)
#study = optuna.load_study(study_name="hyper-parameters-sear", storage=storage_name)
study.optimize(func=obj, n_trials=10)

d={}
print("Best parameters")

for key, value in study.best_trial.params.items():
    d[key]=value
    print(f"    {key}: {value}")
with open('/home/navya/gpu/cv_attr/cv_f1_attr/cv_300_ent_attr/final_params.json', 'w') as f:
    json.dump(d, f)


joblib.dump(study, "/home/navya/gpu/cv_attr/cv_f1_attr/cv_300_ent_attr/study.pkl")
