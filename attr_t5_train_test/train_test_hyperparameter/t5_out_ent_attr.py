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
import warnings
warnings.filterwarnings("ignore")
import random


with open('/home/navya/gpu/data/train_data_out_ent_attr.json') as jsonfile:
    train_data = json.load(jsonfile)


with open('/home/navya/gpu/data/val_data_out_ent_attr.json') as jsonfile:
    val_data = json.load(jsonfile)

print(f"Total Training dataset samples : {len(train_data)}")

print(f"Total validation set samples : {len(val_data)}")

print(f"sample training data :{train_data[0]}")
print(f"sample validation data : {val_data[0]}")

df_train = pd.DataFrame(train_data)
df_val = pd.DataFrame(val_data)

def preprocess(row,max_input_len,max_target_len,tok):
    tok=tok
    input_str = row['input']
    target_str = row['output']
    tokenized_inputs = tok(input_str, max_length=max_input_len, truncation=True)
    tokenized_targets = tok(text_target=target_str, max_length=max_target_len, truncation=True)
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
#load model parameters
with open("/home/navya/gpu/ent_attr/final_params_ent_attr.json") as f:
  params=json.load(f)
print(f"Parameters : {params}")

model_name = "t5-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name,dropout_rate=params['dropout_rate'])
tokenizer.add_tokens(["<sep>"])
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

train_dataset = df_train.apply(lambda row: preprocess(row,params['input_seq_length'],params['output_seq_length'],tokenizer), axis=1)
val_dataset = df_val.apply(lambda row: preprocess(row, params['input_seq_length'],params['output_seq_length'],tokenizer), axis=1)
train_dataset=train_dataset.to_list()
val_dataset=val_dataset.to_list()
train_dataset = MyDataset(train_dataset)
val_dataset = MyDataset(val_dataset)

dir='/home/navya/gpu/ent_attr/train_ent_attr/checkpoints/'

training_args = Seq2SeqTrainingArguments(
  
  output_dir=dir,
  evaluation_strategy="epoch",
  save_strategy="epoch",
   logging_strategy='epoch',
  learning_rate=params['learning_rate'],
  per_device_train_batch_size=params['per_device_train_batch_size'],
  per_device_eval_batch_size=params['per_device_eval_batch_size'],
  weight_decay=params['weight_decay'],
  save_total_limit=1,
  num_train_epochs=params['num_train_epochs'],
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
print(f"Evaluation result : {eval_result}")
print(f"Evaluation loss : {eval_result['eval_loss']}")

#save the pretrained model 
model.save_pretrained('/home/navya/gpu/ent_attr/train_ent_attr/t5_ent_attr_model')
#save tokenizer
tokenizer.save_pretrained('/home/navya/gpu/ent_attr/train_ent_attr/t5_ent_attr_tokenizer')
