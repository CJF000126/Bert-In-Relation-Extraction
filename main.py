#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : 主函数，用于进行训练和测试
@Author             : jifei
@version            : 1.0
'''


import logging
import numpy as np
from pytorch_transformers import WarmupLinearSchedule
from seqeval.metrics import recall_score
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW, BertConfig, BertTokenizer
from loader import load_data, map_id_rel
import random

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
params = {
    'train_batch_size': 4,  # 训练批次大小
    'train_path': 'train.json',  # 训练数据路径
    'eval_batch_size': 32,  # 评估批次大小
    'eval_path': 'dev.json',  # 评估数据路径
    'num_att_heads': 8,  # 多头注意力的头数
    'dropout_rate': 0.1,  # 丢弃率
    # 'num_embed_dim': 128,  # 嵌入维度
    'max_seq_num': 128,  # 最大序列长度
    'bert_path': 'bert-base-chinese',  # BERT模型路径
    'num_epochs': 10,  # 训练轮数
    'learning_rate': 2e-5,  # 学习率
}

# 映射关系ID到关系
rel2id, id2rel = map_id_rel()
logger.info(f"关系数量: {len(rel2id)}")  # 打印关系数量
logger.info(f"关系ID映射: {id2rel}")  # 打印关系ID映射


# 设置随机种子，以确保结果可重复
def setup_seed(seed):
    torch.manual_seed(seed)  # 设置CPU种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的种子
    np.random.seed(seed)  # 设置numpy种子
    random.seed(seed)  # 设置Python随机种子
setup_seed(44)

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()  # 检查是否可用GPU
if USE_CUDA:
    logger.info("使用GPU进行训练")




# 获取模型
def get_model():
    from model import BERT_Classifier  # 导入自定义的BERT分类器
    model = BERT_Classifier(params['bert_path'], len(rel2id), num_heads=params['num_att_heads']).to(device)  # 将模型移动到设备
    return model


model = get_model()

# 初始化分词器和关系映射
tokenizer = BertTokenizer.from_pretrained(params['bert_path'])
train_dataloader = load_data(params['train_path'], tokenizer, params['max_seq_num'], params['train_batch_size'], rel2id)
dev_dataloader = load_data(params['eval_path'], tokenizer, params['max_seq_num'], params['eval_batch_size'], rel2id)


# 评估函数
def eval(net, dataloader):
    net.eval()  # 设置为评估模式
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 不计算梯度
        for batch in tqdm(dataloader, desc="评估中"):
            text, mask, labels = batch  # 从dataloader中获取数据
            if USE_CUDA:
                text, mask, labels = text.cuda(), mask.cuda(), labels.cuda()

            # 前向传播
            outputs = net(text, mask)
            logits = outputs[1]  # 获取logits，假设模型返回的是 (loss, logits)

            # 获取预测结果
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total

    # 计算精确率、召回率和F1得分
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)

    logger.info(f"精确率: {precision:.4f}, F1分数: {f1:.4f}, 召回率: {recall:.4f}")  # 打印准确率、F1得分和召回率
    return acc, f1


# 训练函数
def train(net, train_dataloader, dev_dataloader, num_epochs, learning_rate):
    optimizer = AdamW(net.parameters(), lr=learning_rate)  # 使用AdamW优化器
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=(len(train_dataloader) * num_epochs * 0.05),
                                     t_total=len(train_dataloader) * num_epochs)

    best_acc = 0
    for epoch in range(num_epochs):
        net.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_dataloader, desc=f"训练中 - Epoch {epoch + 1}/{num_epochs}"):
            text, mask, labels = batch  # 从DataLoader中获取数据
            if USE_CUDA:
                text, mask, labels = text.cuda(), mask.cuda(), labels.cuda()

            optimizer.zero_grad()  # 清空梯度
            outputs = net(text, mask, labels)  # 前向传播
            loss = outputs[0]  # 假设模型返回 (loss, logits)
            logits = outputs[1]

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        epoch_acc = correct / total
        epoch_loss = running_loss / len(train_dataloader)

        logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # 评估模型
        acc, f1 = eval(net, dev_dataloader)

        # 保存表现最好的模型
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), f"best_model_{acc:.4f}.pth")


# 开始训练
train(model, train_dataloader, dev_dataloader, params['num_epochs'], params['learning_rate'])
