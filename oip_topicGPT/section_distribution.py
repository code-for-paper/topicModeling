import pandas as pd

src_path = 'data/input/he5r9gsnqdzrdstn.csv'
assign_path = 'data/output/regression_input.csv'

src_df = pd.read_csv(src_path)

cik_gsector = {}

# Method 1: Remove duplicates and keep only the first valid value
for _, row in src_df.iterrows():
    if pd.isna(row['gsector']) or row['gsector'] == '' or pd.isna(row['cik']):
        continue
    # Only add when cik doesn't exist to avoid duplicates
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

# Create a df table with columns: gsector, obs, and all topic columns
df = pd.DataFrame(columns=['gsector', 'obs'] + topics)

for id,ind in enumerate(industry):
    ind_df = assign_df[assign_df['gsector'] == ind]
    # Count the number of each topic in this industry
    # print(len(ind_df))
    obs = len(ind_df)
    
    for topic in topics:
        topic_df = ind_df[ind_df[topic] > 0]
        # Count the number of each topic in this industry
        print(topic, len(topic_df) / obs_all)
        df.loc[id, topic] = len(topic_df) / obs_all
    df.loc[id, 'gsector'] = ind
    df.loc[id, 'obs'] = obs

    # break
    
    # break
    
# Add summary row
total_row_id = len(industry)
df.loc[total_row_id, 'gsector'] = 'All sectors'
df.loc[total_row_id, 'obs'] = df['obs'].sum()  # Sum of obs across all industries

# Calculate sum of all topic columns
for topic in topics:
    df.loc[total_row_id, topic] = df[topic].sum()
    
print(df)

# Round numeric columns to 3 decimal places
numeric_columns = topics + ['obs']  # Numeric columns to format
for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].round(3)

df.to_csv('./data/paper/topic_oip_distribution.csv',index=False)