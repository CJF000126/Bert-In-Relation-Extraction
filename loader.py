#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : 数据加载器和数据处理
@Author             : jifei
@version            : 1.0
'''
import json
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


def setup_seed(seed):
    # 设置随机种子以确保结果可重复
    torch.manual_seed(seed)  # CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 所有GPU的随机种子
    np.random.seed(seed)  # numpy随机种子
    random.seed(seed)  # Python随机种子


setup_seed(44)


def prepare_data():
    # 处理训练和验证数据
    print("---Regenerate Data---")
    with open("./data/DUIE/train_data.json", 'r', encoding='utf-8') as load_f:
        info = []
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data = {
                    'rel': j["predicate"],  # 关系
                    'ent1': j["object"],  # 实体1
                    'ent2': j["subject"],  # 实体2
                    'text': dic['text']  # 文本
                }
                info.append(single_data)

    # 将处理后的训练数据保存到 train.json
    with open("./data/DUIE/train.json", "w", encoding='utf-8') as dump_f:
        for i in info:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")

    # 处理验证数据
    with open("./data/DUIE/dev_data.json", 'r', encoding='utf-8') as load_f:
        info = []
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data = {
                    'rel': j["predicate"],  # 关系
                    'ent1': j["object"],  # 实体1
                    'ent2': j["subject"],  # 实体2
                    'text': dic['text']  # 文本
                }
                info.append(single_data)

    # 将处理后的验证数据保存到 dev.json
    with open("data/DUIE/dev.json", "w", encoding='utf-8') as dump_f:
        for i in info:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")


def map_id_rel():
    # 映射关系ID到关系名
    id2rel = {
        0: 'UNK',
        1: '主演',
        2: '歌手',
        3: '简称',
        4: '总部地点',
        5: '导演',
        6: '出生地',
        7: '目',
        8: '出生日期',
        9: '占地面积',
        10: '上映时间',
        11: '出版社',
        12: '作者'
    }
    rel2id = {rel: i for i, rel in id2rel.items()}  # 创建关系到ID的映射
    return rel2id, id2rel


class loadDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, rel2id):
        self.data = []
        self.tokenizer = tokenizer  # BERT分词器
        self.max_length = max_length  # 最大序列长度
        self.rel2id = rel2id  # 关系ID映射
        # 从JSON文件加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                dic = json.loads(line)
                self.data.append(dic)

    def __len__(self):
        return len(self.data)  # 返回数据集大小

    def __getitem__(self, idx):
        dic = self.data[idx]
        # 准备标签
        label = self.rel2id.get(dic['rel'], 0)  # 获取关系ID，默认0为'UNK'
        # 准备输入句子，连接实体和文本
        sentence = dic['ent1'] + self.tokenizer.sep_token + dic['ent2'] + self.tokenizer.sep_token + dic['text']

        # 分词处理
        encoded = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,  # 添加特殊标记
            max_length=self.max_length,
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 超过最大长度时截断
            return_attention_mask=True,  # 返回注意力掩码
            return_tensors='pt'  # 返回PyTorch张量
        )

        input_ids = encoded['input_ids'].squeeze(0)  # 移除批次维度
        attention_mask = encoded['attention_mask'].squeeze(0)  # 移除批次维度
        return input_ids, attention_mask, torch.tensor(label)  # 返回输入ID、注意力掩码和标签


# 加载数据的函数，使用DataLoader进行批处理
def load_data(file_path, tokenizer, max_length, batch_size, rel2id):
    dataset = loadDataset(file_path, tokenizer, max_length, rel2id)  # 创建数据集实例
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 创建DataLoader
    return dataloader
