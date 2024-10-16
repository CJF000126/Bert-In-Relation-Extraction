#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : 模型定义在这里。使用BERT作为骨干
@Author             : jifei
@version            : 1.0
'''
from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class BERT_Classifier(nn.Module):


    def __init__(self, bert_path, label_num, num_heads=8):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_path)
        # 多头注意力机制，增强模型的表示能力
        self.multihead_attn = nn.MultiheadAttention(embed_dim=768, num_heads=num_heads)
        self.dropout = nn.Dropout(0.1)  # Dropout层，减少过拟合
        self.norm = nn.LayerNorm(768)  # 层归一化
        self.activation = nn.GELU()  # GELU激活函数
        self.fc = nn.Linear(768, label_num)  # 全连接层，用于分类
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def forward(self, x, attention_mask, label=None):
        # BERT编码器输出
        encoder_output = self.encoder(x, attention_mask=attention_mask).last_hidden_state

        # # 调整输出维度以适应多头注意力机制
        encoder_output = encoder_output.permute(1, 0,
                                                2)  # 从 (batch_size, seq_length, embed_dim) 转换为 (seq_length, batch_size, embed_dim)

        #自注意力机制
        attn_output, _ = self.multihead_attn(
            encoder_output,
            encoder_output,
            encoder_output,
            key_padding_mask=(1 - attention_mask).bool()  # 根据 attention_mask 来生成 padding mask
        )

        # 还原维度为 (batch_size, seq_length, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)

        # print(f"attn_output:{attn_output.size()}")
        # 提取 [CLS] token 的表示，通常位于序列的第一个位置
        x = attn_output[:, 0, :]  # 选择第一个 token 作为句子级别的表示

        # Dropout层，减少过拟合
        x = self.dropout(x)

        # # 加入层归一化和激活函数
        x = self.norm(x)
        x = self.activation(x)

        # 全连接层用于分类
        x = self.fc(x)

        # 如果没有提供标签，则返回分类结果
        if label is None:
            return None, x
        else:
            # 计算损失并返回
            return self.criterion(x, label), x




