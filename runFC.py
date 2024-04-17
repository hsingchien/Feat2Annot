#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import argparse
import time


from Feat2Annot import Feat2AnnotFCModel
from util import PoseDatasetDiscrete, prepare_dataset_discrete
from typing import List, Tuple, Dict, Set, Union
import torch.nn.utils
from torch.utils.data import DataLoader, WeightedRandomSampler
from torcheval import metrics


argp = argparse.ArgumentParser()
argp.add_argument(
    "--path",
    default="./data"
)
argp.add_argument(
    "--nepochs",
    default=250
)
argp.add_argument(
    "--train_size",
    default=0.8,
)
argp.add_argument(
    "--batch_size",
    default=512,
)
argp.add_argument(
    "--read_params_path",
    default=None,
)
argp.add_argument(
    "--write_params_path",
    default="fcparams.param",
)
argp.add_argument(
    "--output_path", 
    default=None,
    )
argp.add_argument(
    "--log_every", 
    default=300
    )
argp.add_argument(
    "--val_every",
    default=20000,
)
args = argp.parse_args()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dataset = prepare_dataset_discrete(args.path,device=device)
train_proportion = float(args.train_size)
train_batch_size = int(args.batch_size)
batch_size = int(args.batch_size)
log_every = int(args.log_every)
params_write_path = str(args.write_params_path)
params_read_path = str(args.read_params_path)
val_every = int(args.val_every)
nepoch = int(args.nepochs)

generator1 = torch.Generator().manual_seed(42)

sample_weight = dataset.get_sample_weight()

train_data, val_data = torch.utils.data.random_split(dataset, (train_proportion, 1-train_proportion),generator1)
train_idx = train_data.indices
val_idx = val_data.indices


train_sampler = WeightedRandomSampler(weights=sample_weight[train_idx], num_samples=len(train_idx),replacement=True)
val_sampler = WeightedRandomSampler(weights=sample_weight[val_idx], num_samples=len(val_idx),replacement=True)

train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
val_dataloader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler)

# for i in range(10):
#     _,b = next(train_dataloader)
#     _,a = next(val_dataloader)
#     _,countsb = torch.unique(b,return_counts=True)
#     _,countsa = torch.unique(a,return_counts = True)
#     print(countsb/len(b), countsa/len(a))
    
    

print(dataset.get_annot_class())
hidden_size = [2048,2048,2048]
model = Feat2AnnotFCModel(
    input_size = dataset._num_feature,
    hidden_size=hidden_size,
    target_class=dataset.get_annot_class()
)
model = model.to(device)
metric = metrics.MulticlassAccuracy()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epoch = train_iter = log_iter = val_iter = cum_loss = total_iter = 0
model.train()
loss = torch.nn.CrossEntropyLoss()
begin_time = time.time()
while epoch+1<=nepoch:
    epoch+=1
    for source_feature, tgt_annot in train_dataloader:
        train_iter += 1
        val_iter += 1
        total_iter += 1
        optimizer.zero_grad()
        logits = F.log_softmax(model(source_feature),dim=-1)
        celoss = loss(logits, tgt_annot.squeeze(1))
        celoss = celoss.sum()/train_batch_size
        celoss.backward()
        optimizer.step()
        celoss_val = celoss.item()
        cum_loss += celoss_val
        if train_iter >= log_every:
            log_iter += 1
            print(f"Epoch {epoch}, cumulative loss {cum_loss}, time per {log_every} iter {(time.time()-begin_time)/log_iter}")
            train_iter = 0
            cum_loss = 0
        if val_iter >= val_every:
            print(f"valuation at epoch {epoch} iter {total_iter}")
            model.eval()
            metric.reset()
            hat_count = torch.zeros((dataset.get_annot_class()["class"],), device=device,dtype=torch.int64)
            for source_feature,tgt_annot in tqdm(val_dataloader):
                logits = model(source_feature)
                annot_hat = torch.argmax(logits,dim=-1)
                metric.update(annot_hat,tgt_annot.squeeze(1))
                labels,count = torch.unique(annot_hat, return_counts=True)
                temp_count = torch.zeros_like(hat_count,device=device)
                temp_count[labels] = count
                hat_count+=temp_count
            val_metric = metric.compute()
            print(f"validation accuracy {val_metric}")
            print(hat_count)
            metric.reset()
            model.train()
            val_iter = 0