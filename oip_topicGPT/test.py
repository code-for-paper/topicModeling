
import json
import numpy as np

# Statistics on form length information in ./data/input/10k_filtered.jsonl
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

# Statistics: max, min, mean, std, total length, form count
max_len = max(len_list)
min_len = min(len_list)
avg_len = np.mean(len_list)
std_len = np.std(len_list)

total_len = np.sum(len_list)
form_num = len(len_list)

print(f"Max length: {max_len}")
print(f"Min length: {min_len}")
print(f"Average length: {avg_len}")
print(f"Standard deviation: {std_len}")
print(f"Total length: {total_len/1e6}M characters")
print(f"Form count: {form_num}")
print(f'Valid cik: {cik_count}')
