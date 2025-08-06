import pandas as pd

src_path = 'data/input/he5r9gsnqdzrdstn.csv'
assign_path = 'data/output/regression_input.csv'

src_df = pd.read_csv(src_path)

cik_gsector = {}

# 方法1：去重并只保留第一个有效值
for _, row in src_df.iterrows():
    if pd.isna(row['gsector']) or row['gsector'] == '' or pd.isna(row['cik']):
        continue
    # 只有当cik不存在时才添加，避免重复
    if row['cik'] not in cik_gsector:
        cik_gsector[row['cik']] = int(row['gsector'])

assign_df = pd.read_csv(assign_path)
assign_df['gsector'] = assign_df['cik'].map(cik_gsector)

industry = sorted(set(cik_gsector.values()))

topics = [ "Supplier & Manufacturing Collaboration",
    "Customer & Community Engagement",
    "Strategic & Joint Partnerships",
    "IP & Technology Licensing",
    "Industry-Academia & Research Collaboration",
    "Technology & Platform Ecosystems",
    "Open Innovation & Ecosystem Strategy",
    "Marketing & Commercialization Cooperation",'OIP']

obs_all = len(assign_df)

# 做一个df表  列有 gsector obs topics所有列
df = pd.DataFrame(columns=['gsector', 'obs'] + topics)

for id,ind in enumerate(industry):
    ind_df = assign_df[assign_df['gsector'] == ind]
    # 统计各个topic在该行业的数量
    # print(len(ind_df))
    obs = len(ind_df)
    
    for topic in topics:
        topic_df = ind_df[ind_df[topic] > 0]
        # 统计各个topic在该行业的数量
        print(topic, len(topic_df) / obs_all)
        df.loc[id, topic] = len(topic_df) / obs_all
    df.loc[id, 'gsector'] = ind
    df.loc[id, 'obs'] = obs

    # break
    
    # break
    
# 添加汇总行
total_row_id = len(industry)
df.loc[total_row_id, 'gsector'] = 'All sectors'
df.loc[total_row_id, 'obs'] = df['obs'].sum()  # 所有行业的obs加和

# 计算所有topic列的加和
for topic in topics:
    df.loc[total_row_id, topic] = df[topic].sum()
    
print(df)

# 将数值列保留3位小数
numeric_columns = topics + ['obs']  # 需要格式化的数值列
for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].round(3)

df.to_csv('./data/paper/topic_oip_distribution.csv',index=False)