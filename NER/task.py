# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:09:20 2024

@author: gimal
"""


import pytorch_lightning as pl
import pandas as pd
import torch
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from transformers import BertTokenizerFast
from transformers import BertPreTrainedModel
from seqeval.metrics import f1_score, accuracy_score

from utils import Config


#로그용 정확도 계산 함수.
def accuracy(preds,labels,ignore_index=None):
  #no_grad는 gradient를 트래킹하지 않도록 해서 메모리 효율을 높여주는거임.
  with torch.no_grad():
    assert preds.shape[0] == len(labels)
    correct = torch.sum(preds == labels)
    total = torch.sum(torch.ones_like(labels))
    if ignore_index is not None:
      correct -= torch.sum(torch.logical_and(preds == ignore_index, preds == labels))
      # accuracy의 분모 가운데 ignore index에 해당하는 것 제외
      total -= torch.sum(labels == ignore_index)
  return correct.to(dtype=torch.float) / total.to(dtype=torch.float)





#트레이너에 들어갈 task
class NERTask(pl.LightningModule):
  #model이랑 arguments 받아서 이니셜라이징
  def __init__(self,model : BertPreTrainedModel,args : Config):
    super().__init__()
    self.model = model
    self.args = args
    self.tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
  # optimizer는 AdamW사용
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr = self.args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    return {
        "optimizer" : optimizer,
        "scheduler" : scheduler,
    }

  def training_step(self,inputs,batch_idx):
    outputs = self.model(**inputs)
    #예측값이랑 레이블 이용해서 정확도 계산
    preds = outputs.logits.argmax(dim=-1)
    labels = inputs["labels"]
    acc = accuracy(preds,labels,ignore_index = -100)
    #로그에 찍음.
    self.log("loss", outputs.loss, prog_bar = True, logger= True, on_step=True, on_epoch = False)
    self.log("acc", acc, prog_bar=True,logger=True,on_step=True,on_epoch=False)
    return outputs.loss


  def test_step(self,inputs,batch_idx):
    outputs = self.model(**inputs)
    preds = outputs.logits.argmax(dim = -1)

    all_token = []
    all_token_predictions = []
    all_token_labels = []
    id2label = [
        'B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O'
    ]
    labels = inputs["labels"]
    input_ids = inputs["input_ids"]

    #텍스트 받음
    for tokens in input_ids:
      filtered_token = []
      for i in range(len(tokens)):
        text = self.tokenizer.convert_ids_to_tokens(tokens[i].tolist())
        if text == "[CLS]":
          continue
        elif text == "[PAD]":
          continue
        elif text == "[SEP]":
          continue
        else:
          filtered_token.append(text)
      all_token.append(filtered_token)

    #예측토큰이랑, 실제 정답 토큰 받음.
    token_predictions = preds.detach().cpu().numpy()
    for token_prediction, label in zip(token_predictions,labels):
      filtered = []
      filtered_label = []
      for i in range(len(token_predictions)):
        if label[i].tolist() == -100:
          continue
        filtered.append(id2label[token_prediction[i]])
        filtered_label.append(id2label[label[i].tolist()])
      all_token_predictions.append(filtered)
      all_token_labels.append(filtered_label)

    df = pd.DataFrame({"text": all_token, "labels":all_token_labels, "preds":all_token_predictions})

    #최초 생성이면 w
    if not os.path.exists("test_result.csv"):
      df.to_csv("test_result.csv",mode = "w",index=False,sep = ",")
    else:
      df.to_csv("test_result.csv",mode = "a",index=False,sep = ",")

    acc = accuracy(preds,labels,ignore_index = -100)
    self.log("test_loss", outputs.loss, prog_bar = True, logger= True, on_step=True, on_epoch = False)
    self.log("test_acc", acc, prog_bar=True,logger=True,on_step=True,on_epoch=False)

    F1_score = f1_score(all_token_labels, all_token_predictions, average="macro")
    accuracy_sco = accuracy_score(all_token_labels, all_token_predictions)

    return {"loss" : outputs.loss, "accuracy" : float(accuracy_sco), "F1_score" : float(F1_score) }

  def validation_step(self,inputs,batch_idx):
    outputs = self.model(**inputs)
    preds = outputs.logits.argmax(dim=-1)
    labels = inputs["labels"]
    acc = accuracy(preds,labels,ignore_index = -100)
    self.log("val_loss", outputs.loss, prog_bar = False, logger= True, on_step=False, on_epoch = True)
    self.log("val_acc", acc, prog_bar=True,logger=True,on_step=False,on_epoch=True)
    return outputs.loss