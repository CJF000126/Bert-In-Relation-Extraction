# Bert In Relation Extraction

ps1:这是一个基于Bert+Attention的模型，用于关系抽取。

ps2:基于原作者项目上进行修改，优化了DataLoad加入了Attention、归一化等。
原作者gitHub地址：https://github.com/Ricardokevins/Bert-In-Relation-Extraction


# 效果：

```
Source Text:  《在夏天冬眠》是容祖儿演唱的一首歌曲，收录于专辑《独照》中
Entity1:  独照  Entity2:  在夏天冬眠  Predict Relation:  所属专辑  True Relation:  所属专辑


Source Text:  2.花千骨花千骨是由慈文传媒集团制作并发行，高林豹、林玉芬、梁胜权联合执导，霍建华 、赵丽颖领衔主演，蒋欣、杨烁、张丹峰、马可、徐海乔、李纯等主演的古装仙侠玄幻 仙侠剧
Entity1:  赵丽颖  Entity2:  花千骨  Predict Relation:  主演  True Relation:  主演


Source Text:  在与王祖贤恋爱期间的齐秦，也是他事业最辉煌的时期，《大约在冬季》《无情的雨》《不让我的眼泪陪我过夜》《如果云知道》《夜夜夜夜》等经典曲目都是他为王祖贤所创作的，从这些歌也能感受到两个人是真爱，但是为什么就是没有一个结果呢
Entity1:  齐秦  Entity2:  大约在冬季  Predict Relation:  歌手  True Relation:  歌手


Source Text:  《甜蜜与厮杀》是连载在红袖添香网上的一部奇幻魔法小说，作者是kijimi1
Entity1:  kijimi1  Entity2:  甜蜜与厮杀  Predict Relation:  作者  True Relation:  作者
```


# 使用方法

## 准备

1. 将DUIE文件路径放置于代码同目录（或者自己的数据，具体可见loader.py)，更加具体的获取和数据处理见下文
2. 将bert-base-chinese放置于同目录下的bert-base-chinese下或者自行指定位置
3. 安装对应的包

## train and eval

（注意，在此之前，请做好数据的获取和预处理，步骤见文）

（注意：需要手动设置loader.py实体关系，为了测试我只提取了12类关系）

**python3 main.py**执行训练，并得到Fine-Tuing后的BERT

**python3 test.py**得到测试输出，或自行阅读代码，修改test函数的传入参数内容即可自定义。




# 数据

数据使用的是百度发布的DUIE数据，包含了实体识别和关系抽取

旧链接：

原数据地址：https://ai.baidu.com/broad/download?dataset=dureader

**由于百度可能重新组织了数据集的页面，所以旧链接似乎用不了了，我去AIStduio里找了一个我印象里差不多的（关键词是DUIE，Knowledge Extraction）
你们可以根据自己的需要去找自己喜欢的数据集**

新链接：https://aistudio.baidu.com/aistudio/datasetdetail/88472


打开后在左侧栏选择knowledge extraction，然后如下界面点击下载train_data.json和dev_data.json，然后放到对应的位置

**运行loader.py里的prepare_data**，观察到目录里生成了**train.json和dev.json**


截止这里，数据的预处理完成了，可以运行main和demo

![Inkedimage-20210204112312401_LI](README.assets/Inkedimage-20210204112312401_LI.jpg)



我对数据进行了预处理，提取关系抽取需要的部分

关系设定有49类，还是非常的丰富的

```
id2rel={0: 'UNK', 1: '主演', 2: '歌手', 3: '简称', 4: '总部地点', 5: '导演', 
        6: '出生地', 7: '目', 8: '出生日期', 9: '占地面积', 10: '上映时间',
        11: '出版社', 12: '作者', 13: '号', 14: '父亲', 15: '毕业院校', 
        16: '成立日期', 17: '改编自', 18: '主持人', 19: '所属专辑', 
        20: '连载网站', 21: '作词', 22: '作曲', 23: '创始人', 24: '丈夫', 
        25: '妻子', 26: '朝代', 27: '民族', 28: '国籍', 29: '身高', 30: '出品公司', 
        31: '母亲', 32: '编剧', 33: '首都', 34: '面积', 35: '祖籍', 36: '嘉宾', 
        37: '字', 38: '海拔', 39: '注册资本', 40: '制片人', 41: '董事长', 42: '所在城市',
        43: '气候', 44: '人口数量', 45: '邮政编码', 46: '主角', 47: '官方语言', 48: '修业年限'}   
    
```

数据的格式如下，ent1和ent2是实体，rel是关系

![image.png](figure/image-1603561010980.png)



# Model

模型就是直接使用Bert用于序列分类的（BertEncoder+Attention+Fc+CrossEntropy）

具体的处理就是把ent1，ent2和sentence直接拼接送进模型


# Result

从百度的原数据中选择20000条，测试数据2000条（原数据相对很小的一部分）

训练参数：10 Epoch，0.001学习率，设置label共有49种（包含UNK，代表新关系和不存在关系）

然后在训练前和训练后的分别在测试数据上测试，可以看到**Fine-Tuing**高度有效


![image.png](figure/image-1603561010979.png)



# 实际测试

在数据中抽取一部分实际测试

效果不错

![image.png](figure/image-1603561010074.png)

![image.png](figure/image.png)




