import numpy as np
import pandas as pd

raw_table_path = './data/input/he5r9gsnqdzrdstn.csv'
stat_path = './data/output/stat.jsonl'

# raw_table = pd.read_csv(raw_table_path,dtype={'cik':str})
# df = raw_table[['tic', 'cik', 'mkvalt',
#                        'datadate', 'dltt', 'dlc', 'at']]

df = pd.read_csv(raw_table_path, dtype={'cik': str})

# Select necessary fields
required_cols = [
    'cik', 'fyear', 'datadate', 'at', 'lt', 'ni', 'xrd', 'sale','ppent',
    'prcc_f', 'csho'  # For calculating Tobin's Q
]
df = df[required_cols].copy()

# Replace invalid values (like 0) to avoid division by zero errors
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with any key columns being null (strict filtering)
df.dropna(subset=required_cols, inplace=True)

# Construct variables
df['log_at'] = np.log(df['at']) # Company size (asset-based log scale indicator) Control variable: control company size impact on Tobin's Q
# df['leverage'] = df['lt'] / df['at'] # Debt-to-asset ratio (total debt as proportion of total assets) Control variable: reflects company financial leverage
df['leverage'] = df['ppent'] / df['at'] # Capital intensity
df['roa'] = df['ni'] / df['at'] # Return on assets (net income divided by total assets) Control variable: measures company profitability
df['xrdint'] = df['xrd'] / df['sale'] # R&D -> ✔ Can be used as explanatory variable (to test R&D's own impact on Tobin's Q) ✔ Or interaction with OIP: OIP × xrdint (test complementarity)
df['xrdint'] = np.log1p(df['xrdint'])  # If subsequent model standardization is more appropriate
df['tobinq'] = (df['prcc_f'] * df['csho'] + df['lt']) / df['at'] # Tobin's Q value, measures enterprise market value premium relative to book assets Dependent variable: used to measure enterprise performance

# Keep final fields for regression
final_cols = ['cik', 'fyear', 'datadate', 'log_at', 'leverage', 'roa', 'xrdint', 'tobinq']
regression_df = df[final_cols].copy()

# View first few rows
print(regression_df.head())

# Optional: save processed data
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
                # Multiple topics
                score_list = forms[date]['score']
                
                # refer date
                refer_date = forms[date]['refer_date']
                refer_tobins_q = forms[date]['refer_tobins_q']
                oip_score = forms[date]['oip_score']
                stat_dict[cik][refer_date] = {
                    'date':date,
                    'refer_tobins_q':refer_tobins_q,
                    'score_list':score_list,
                    'oip_score':oip_score
                }

# print(f'Total {cnt} rows')

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

# Create empty pandas csv table
result_df = pd.DataFrame(columns=columns
                        )

topic_index = {topic: i for i, topic in enumerate(topic_list)}

# Iterate through df
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
            # Organize topic score list in topic_list order, set missing ones to empty, then merge lists
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
            # Add one row
            refer_tobins_q = form_dict[datadate]['refer_tobins_q']
            form_date = form_dict[datadate]['date']
            
            result_df.loc[real_cnt] = [cik, fyear, datadate, log_at, leverage, roa, xrdint, tobinq]+ [refer_tobins_q,form_date] + list(index_score.values())
            real_cnt += 1

print(result_df)
print(f'{real_cnt} valid rows')

result_df.to_csv('./data/output/regression_input_example.csv', index=False)
