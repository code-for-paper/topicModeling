
import json
import numpy as np

# 统计./data/input/10k_filtered.jsonl的form的长度信息
corpus_path = './data/input/10k_filtered.jsonl'
# corpus_path = '/root/autodl-tmp/10k.jsonl'

len_list = []
cik_count = 0
with open(corpus_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        forms = data['10-k']
        if len(forms) > 0:
            cik_count += 1
        for item in forms:
            text = item['form']
            # len_list.append( min( len(text),5000) )
            len_list.append( len(text)  )

# 统计最大值 最小值 平均值 标准差 总长度 form数量
max_len = max(len_list)
min_len = min(len_list)
avg_len = np.mean(len_list)
std_len = np.std(len_list)

total_len = np.sum(len_list)
form_num = len(len_list)

print(f"最大长度: {max_len}")
print(f"最小长度: {min_len}")
print(f"平均长度: {avg_len}")
print(f"标准差: {std_len}")
print(f"总长度: {total_len/1e6}M字符")
print(f"form数量: {form_num}")
print(f'valid cik:{cik_count}')
