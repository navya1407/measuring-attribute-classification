#importing necesssary libraries
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

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(20)
with open('/home/navya/gpu/ent_attr/val_data_6k_out_ent_attr.json') as jsonfile:
    val_data = json.load(jsonfile)
with open('/home/navya/gpu/ent_attr/train_data_40k_out_ent_attr.json') as jsonfile:
    train_data = json.load(jsonfile)



# Split the data into training, validation, and testing sets
#val_data = val_data[:2000]
#train_data = train_data[2000:]

df_train = pd.DataFrame(train_data)
df_val = pd.DataFrame(val_data)




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




def obj(trial: optuna.Trial):
  learning_rate=trial.suggest_loguniform("learning_rate", low=4e-5, high=0.01)
  weight_decay=trial.suggest_loguniform("weight_decay", 4e-5, 0.01)
  train_epochs=trial.suggest_int("num_train_epochs", low=10, high=100)
  train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [8,16,32,64,128])
  eval_batch_size=trial.suggest_categorical("per_device_eval_batch_size",[8,16,32,64])
  #d_model = trial.suggest_categorical('d_model', [1024])
  #num_layers = trial.suggest_categorical('num_layers', [16])
  input_seq_length = trial.suggest_categorical('input_seq_length', [64,128,256])
  output_seq_length = trial.suggest_categorical('output_seq_length', [6,10,14])
  dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
 
  print(f" Trial number : {trial.number} ----> lr_rate : {learning_rate} , weight_decay : {weight_decay} , epochs : { train_epochs} , train batch size: {train_batch_size} ,eval batch size: {eval_batch_size} , input_seq_length : {input_seq_length}, output_seq_length: {output_seq_length},,dropout_rate: {dropout_rate} " ) 
 
  
  model_name = "t5-base" 
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name,dropout_rate=dropout_rate)# d_model=d_model,num_layers=num_layer,,ignore_mismatched_sizes=True
  tokenizer.add_tokens(["<sep>"])
  model.resize_token_embeddings(len(tokenizer)) 
  model=model.to(device)
  data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
  train_dataset = df_train.apply(lambda row: preprocess(row,input_seq_length,output_seq_length,tokenizer), axis=1)
  val_dataset = df_val.apply(lambda row: preprocess(row, input_seq_length,output_seq_length,tokenizer), axis=1)
  train_dataset=train_dataset.to_list()
  val_dataset=val_dataset.to_list()
  train_dataset = MyDataset(train_dataset)
  val_dataset = MyDataset(val_dataset)
  dir='/home/navya/gpu/ent_attr/hyperparams_ent_attr/'+f"trial_{trial.number}"

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
  load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False
   
    
  
  
)
 
  trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    
    data_collator=data_collator,
    tokenizer=tokenizer
    
) 
  trainer.train()
  eval_result = trainer.evaluate(eval_dataset=val_dataset)
  print(f"{trial.number} -----> loss : { eval_result['eval_loss'] }")
  return eval_result['eval_loss']

# We want to minimize the loss!
sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(study_name="hyper-parameters-searching_ent_attr", direction="minimize",sampler=sampler)

study.optimize(func=obj, n_trials=10)

d={}
print("Best parameters")

for key, value in study.best_trial.params.items():
    d[key]=value
    print(f"    {key}: {value}")

with open('/home/navya/gpu/ent_attr/final_params_ent_attr.json', 'w') as f:
    json.dump(d, f)


joblib.dump(study, "/home/navya/gpu/ent_attr/study_ent_attr.pkl")

