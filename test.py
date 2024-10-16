import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
import torch
import time
import argparse
import json
import os
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig
#from transformers import BertPreTrainedModel
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

setup_seed(44)

from transformers import BertModel

from loader import map_id_rel

rel2id, id2rel = map_id_rel()

print(len(rel2id))
print(id2rel)

USE_CUDA = torch.cuda.is_available()

def get_train_args():
    labels_num=len(rel2id)
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=1,help = '每批数据的数量')
    parser.add_argument('--nepoch',type=int,default=30,help = '训练的轮次')
    parser.add_argument('--lr',type=float,default=0.001,help = '学习率')
    parser.add_argument('--gpu',type=bool,default=True,help = '是否使用gpu')
    parser.add_argument('--num_workers',type=int,default=2,help='dataloader使用的线程数量')
    parser.add_argument('--num_labels',type=int,default=len(id2rel),help='分类类数')
    parser.add_argument('--data_path',type=str,default='./data',help='数据路径')
    opt=parser.parse_args()
    print(opt)
    return opt

def get_model(opt):
    from model import BERT_Classifier  # 导入自定义的BERT分类器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT_Classifier('bert-base-chinese', len(rel2id), num_heads=8).to(device)  # 将模型移动到设备
    return model


def test(net, text_list, ent1_list, ent2_list, result):
    net.eval()
    max_length = 128

    # 加载模型权重
    net.load_state_dict(torch.load('best_model_0.9444.pth'))
    rel_list = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    with torch.no_grad():
        for text, ent1, ent2, label in zip(text_list, ent1_list, ent2_list, result):
            sent = ent1 + tokenizer.sep_token + ent2 + tokenizer.sep_token + text
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            indexed_tokens += [0] * (max_length - len(indexed_tokens))  # 填充
            indexed_tokens = indexed_tokens[:max_length]  # 截断
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)

            att_mask = torch.zeros(indexed_tokens.size()).long()
            att_mask[0, :len(indexed_tokens[0])] = 1

            if USE_CUDA:
                indexed_tokens = indexed_tokens.cuda()
                att_mask = att_mask.cuda()

            outputs = net(indexed_tokens, att_mask)
            # print(outputs)
            loss,logits = outputs
            _, predicted = torch.max(logits, 1)
            # print(predicted)
            result = predicted.cpu().numpy().tolist()[0]

            print("Source Text: ", text)
            print("Entity1: ", ent1, " Entity2: ", ent2, " Predict Relation: ", id2rel[result], " True Relation: ",
                  label)
            print('\n')
            rel_list.append(id2rel[result])

    return rel_list
opt = get_train_args()
model=get_model(opt)

if USE_CUDA:
    model=model.cuda()

from random import choice

text_list=[]
ent1=[]
ent2=[]
result=[]
with open("dev.json", 'r', encoding='utf-8') as load_f:
    lines=load_f.readlines()
    total_num=10
    while total_num>0:
        line=choice(lines)
        dic = json.loads(line)
        text_list.append(dic['text'])
        ent1.append(dic['ent1'])
        ent2.append(dic['ent2'])
        result.append(dic['rel'])
        total_num-=1
        if total_num<0:
            break

test(model,text_list,ent1,ent2,result)
