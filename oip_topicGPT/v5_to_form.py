import numpy as np
import pandas as pd

raw_table_path = './data/input/he5r9gsnqdzrdstn.csv'
stat_path = './data/output/stat.jsonl'

# raw_table = pd.read_csv(raw_table_path,dtype={'cik':str})
# df = raw_table[['tic', 'cik', 'mkvalt',
#                        'datadate', 'dltt', 'dlc', 'at']]

df = pd.read_csv(raw_table_path, dtype={'cik': str})

# 选取必要字段
required_cols = [
    'cik', 'fyear', 'datadate', 'at', 'lt', 'ni', 'xrd', 'sale','ppent',
    'prcc_f', 'csho'  # 用于计算 Tobin's Q
]
df = df[required_cols].copy()

# 替换非法值（如 0），避免除零错误
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 丢弃任何关键列为空的行（严格过滤）
df.dropna(subset=required_cols, inplace=True)

# 构造变量
df['log_at'] = np.log(df['at']) #公司规模（以资产为基础的对数化规模指标） 控制变量：控制公司大小对 Tobin's Q 的影响
# df['leverage'] = df['lt'] / df['at'] # 资产负债率（总负债占总资产的比例）控制变量：反映公司财务杠杆
df['leverage'] = df['ppent'] / df['at'] # Capital intensity 资本密集强度
df['roa'] = df['ni'] / df['at'] #资产收益率（净利润除以总资产） 控制变量：衡量公司盈利能力
df['xrdint'] = df['xrd'] / df['sale'] # R&D -> ✔ 可作为 解释变量（用于检验 R&D 本身对 Tobin's Q 的影响）✔ 或与 OIP 交互项：OIP × xrdint（检验互补性）
df['xrdint'] = np.log1p(df['xrdint'])  # 如果后续模型做标准化更合适
df['tobinq'] = (df['prcc_f'] * df['csho'] + df['lt']) / df['at'] # Tobin's Q 值，衡量企业市场价值相对于账面资产的溢价 被解释变量（因变量）：用来衡量企业绩效

# 保留最终用于回归的字段
final_cols = ['cik', 'fyear', 'datadate', 'log_at', 'leverage', 'roa', 'xrdint', 'tobinq']
regression_df = df[final_cols].copy()

# 查看前几行
print(regression_df.head())

# 可选：保存处理后的数据
# regression_df.to_csv('./data/processed/regression_input.csv', index=False)

stat_dict = {}
import json

# cnt = 0

with open(stat_path,'r') as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            cik = data['cik']
            forms:dict = data['forms']
            # cnt += len(forms.keys())
            stat_dict[cik] = {}
            for date in forms:
                # 有多个topic
                score_list = forms[date]['score']
                
                #refer date
                refer_date = forms[date]['refer_date']
                refer_tobins_q = forms[date]['refer_tobins_q']
                oip_score = forms[date]['oip_score']
                stat_dict[cik][refer_date] = {
                    'date':date,
                    'refer_tobins_q':refer_tobins_q,
                    'score_list':score_list,
                    'oip_score':oip_score
                }

# print(f'共有{cnt}个行')

topic_list = [
    "Supplier & Manufacturing Collaboration",
    "Customer & Community Engagement",
    "Strategic & Joint Partnerships",
    "IP & Technology Licensing",
    "Industry-Academia & Research Collaboration",
    "Technology & Platform Ecosystems",
    "Open Innovation & Ecosystem Strategy",
    "Marketing & Commercialization Cooperation",
    'OIP'
]

columns = ['cik', 'fyear', 'datadate', 
           'log_at', 'leverage', 'roa', 
           'xrdint', 'tobinq',
           'refer_tobins_q','form_date'
          ] + topic_list

# 创建一个空的pandas csv表格
result_df = pd.DataFrame(columns=columns
                        )

topic_index = {topic: i for i, topic in enumerate(topic_list)}

# 遍历df
real_cnt = 0
for index, row in regression_df.iterrows():
    cik = row['cik']
    fyear = row['fyear']
    log_at = row['log_at']
    leverage = row['leverage']
    roa = row['roa']
    xrdint = row['xrdint']
    tobinq = row['tobinq']
    datadate = row['datadate']
    if cik in stat_dict:
        form_dict = stat_dict[cik]
        if datadate in form_dict:
            # 整理topic score列表 按照topic_list顺序 没有的就置为空 然后把列表合并
            score_list = form_dict[datadate]['score_list']
            index_score = {
                id:0 for id in range(len(topic_list))
            }
            
            for item in score_list:
                topic=item['topic']
                score=item['score']
                if topic not in topic_index:
                    continue
                index_score[topic_index[topic]] = score
                
            index_score[topic_index['OIP']] = form_dict[datadate]['oip_score']
            # 加入一行
            refer_tobins_q = form_dict[datadate]['refer_tobins_q']
            form_date = form_dict[datadate]['date']
            
            result_df.loc[real_cnt] = [cik, fyear, datadate, log_at, leverage, roa, xrdint, tobinq]+ [refer_tobins_q,form_date] + list(index_score.values())
            real_cnt += 1

print(result_df)
print(f'有{real_cnt}个合法的行')

result_df.to_csv('./data/output/regression_input.csv', index=False)
