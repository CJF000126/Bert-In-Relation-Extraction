import json
import random

# 定义你需要的12类关系
relations = {0: 'UNK', 1: '主演', 2: '歌手', 3: '简称', 4: '总部地点',
             5: '导演', 6: '出生地', 7: '目', 8: '出生日期', 9: '占地面积',
             10: '上映时间', 11: '出版社', 12: '作者'}

# 输入文件路径
input_file = 'data/DUIE/train.json'

# 输出文件路径
output_file = 'train.json'

# 最大样本数量参数
max_samples = 100  # 可以根据需要修改这个值

# 用于存储已提取的样本
extracted_data = {relation: [] for relation in relations.values()}

# 打开输入文件并逐行处理
with open(input_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        try:
            # 解析每一行的 JSON 数据
            item = json.loads(line.strip())
            rel = item['rel']

            if rel in extracted_data and len(extracted_data[rel]) < max_samples:
                # 只提取每种关系最多max_samples个样本
                extracted_data[rel].append(item)

            # 检查是否所有关系都已提取达到max_samples个样本
            if all(len(extracted_data[relation]) >= max_samples for relation in extracted_data):
                break  # 提前终止循环
        except json.JSONDecodeError as e:
            print(f"解析失败的行: {line.strip()}，错误信息: {e}")

# 将所有提取的样本合并到一个列表中
all_items = []
for relation_items in extracted_data.values():
    all_items.extend(relation_items)

# 打乱顺序
random.shuffle(all_items)

# 将打乱后的数据写入输出文件
with open(output_file, 'w', encoding='utf-8') as f_out:
    for item in all_items:
        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"提取的关系数据已乱序并保存到 {output_file}")
