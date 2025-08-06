import pandas as pd
import statsmodels.api as sm
import numpy as np

# 读取你的准备好的表格
df = pd.read_csv('./data/output/regression_input.csv')

tobinq = 'refer_tobins_q'
tobinq = 'tobinq'

# df['Supplier Collaboration_xrdint'] = df['Supplier Collaboration'] * df['xrdint']
# df['Technology Partnerships_xrdint'] = df['Technology Partnerships'] * df['xrdint']
# OIP 主题变量的平方项（选择几个）
# df['Supplier Collaboration_sq'] = df['Supplier Collaboration'] ** 2
# df['Technology Partnerships_sq'] = df['Technology Partnerships'] ** 2
# df['IP Licensing_sq'] = df['IP Licensing'] ** 2
# df['Strategic Partnerships_sq'] = df['Strategic Partnerships'] ** 2


# 设置解释变量（包含控制变量和 OIP 主题）

explanatory_fixed_vars = [
    'log_at',
    'xrdint',
    'roa',
    'leverage', 
]

explanatory_topic_vars = [
    # "Supplier & Manufacturing Collaboration",
    # "Customer & Community Engagement",
    # "Strategic & Joint Partnerships",
    # "IP & Technology Licensing",
    # "Industry-Academia & Research Collaboration",
    # "Technology & Platform Ecosystems",
    # "Open Innovation & Ecosystem Strategy",
    "Marketing & Commercialization Cooperation",
    # oip sum
    # 'OIP',
]

other_vars = [
 # R&D interaction
    # "Supplier & Manufacturing Collaboration_xrdint",
    # "Customer & Community Engagement_xrdint",
    # "Strategic & Joint Partnerships_xrdint",
    # "IP & Technology Licensing_xrdint",
    # "Industry-Academia & Research Collaboration_xrdint",
    # "Technology & Platform Ecosystems_xrdint",
    # "Open Innovation & Ecosystem Strategy_xrdint",
    # "Marketing & Commercialization Cooperation_xrdint",
    # 'OIP_xrdint'

    # square
    # "Supplier & Manufacturing Collaboration_sq",
    # "Customer & Community Engagement_sq",
    # "Strategic & Joint Partnerships_sq",
    # "IP & Technology Licensing_sq",
    # "Industry-Academia & Research Collaboration_sq",
    # "Technology & Platform Ecosystems_sq",
    # "Open Innovation & Ecosystem Strategy_sq",
    "Marketing & Commercialization Cooperation_sq",
    # 'OIP_sq'
]



# 计算交互项（OIP × xrdint）
for topic in explanatory_topic_vars:
    interaction_col = f"{topic}_xrdint"
    df[interaction_col] = df[topic] * df['xrdint']

# 计算交互项（OIP × roa）
for topic in explanatory_topic_vars:
    interaction_col = f"{topic}_roa"
    df[interaction_col] = df[topic] * df['roa']

# 计算平方项（OIP ^ 2）
for topic in explanatory_topic_vars:
    quadratic_col = f"{topic}_sq"
    df[quadratic_col] = df[topic] ** 2


explanatory_vars = explanatory_fixed_vars + explanatory_topic_vars + other_vars

# 数据清理：处理无穷大和 NaN 值
print(f"原始数据行数: {len(df)}")

# 检查每列的缺失值和无穷大值
for col in [tobinq] + explanatory_vars:
    if col in df.columns:
        inf_count = np.isinf(df[col]).sum()
        nan_count = df[col].isna().sum()
        if inf_count > 0 or nan_count > 0:
            print(f"{col}: {inf_count} 个无穷大值, {nan_count} 个 NaN 值")

# 替换无穷大值为 NaN
df = df.replace([np.inf, -np.inf], np.nan)

# 删除包含 NaN 值的行
df_clean = df[[tobinq] + explanatory_vars].dropna()
# ===================== 剔除尾部影响 ====================
q_low = df_clean[tobinq].quantile(0.02)
q_high = df_clean[tobinq].quantile(0.99)

# 保留正常范围内的样本
df_clean = df_clean[(df_clean[tobinq] >= q_low) & (df_clean[tobinq] <= q_high)]

print(f"清理后数据行数: {len(df_clean)}")
print(f"删除了 {len(df) - len(df_clean)} 行数据")

# 准备数据
X = df_clean[explanatory_vars]
X = sm.add_constant(X)  # 加截距项
y = df_clean[tobinq]         # 因变量

# 最终检查：确保没有 inf 或 NaN
if np.any(np.isinf(X)) or np.any(np.isnan(X)) or np.any(np.isinf(y)) or np.any(np.isnan(y)):
    print("警告：数据中仍然存在 inf 或 NaN 值")
else:
    print("数据清理完成，可以进行回归分析")

# 描述性统计分析
print("描述性统计分析（因变量和解释变量）：")
print(df_clean[[tobinq] + explanatory_vars].describe())

# 描述性统计分析，保存到csv文件
desc_stats = df_clean[[tobinq] + explanatory_vars].describe()
desc_stats.to_csv('./data/output/desc_stats.csv', encoding='utf-8-sig')

print("描述性统计结果已保存到 './data/output/desc_stats.csv'")

# 计算变量间的线性相关系数
print("\n变量间线性相关系数矩阵：")
corr_matrix = df_clean[[tobinq] + explanatory_vars].corr()
print(corr_matrix)

# 显著性检验 双尾 0.05
from scipy.stats import pearsonr

# 👇 强制整个矩阵转换为 string 类型（避免 dtype 冲突）
corr_matrix = corr_matrix.astype(str)

# 用带显著性标记的字符串替换内容，只保存下三角
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        row_name = corr_matrix.index[i]
        col_name = corr_matrix.columns[j]
        
        if i == j:
            # 对角线元素
            corr_matrix.loc[row_name, col_name] = "1.000"
        elif i > j:
            # 下三角元素
            r, p = pearsonr(df_clean[row_name], df_clean[col_name])
            if p < 0.05:
                corr_matrix.loc[row_name, col_name] = f"{r:.3f}*"
            else:
                corr_matrix.loc[row_name, col_name] = f"{r:.3f}"
        else:
            # 上三角元素设为空字符串
            corr_matrix.loc[row_name, col_name] = ""

# 保存相关系数矩阵到csv文件
corr_matrix.to_csv('./data/output/correlation_matrix.csv', encoding='utf-8-sig')
print("相关系数矩阵已保存到 './data/output/correlation_matrix.csv'")

print('-' * 100)



from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pandas as pd

# 提取并标准化解释变量
X = df_clean[explanatory_vars]
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)

# 将标准化后的数组转换为 DataFrame，并保留列名
X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

# 添加常数项
X_scaled_df = sm.add_constant(X_scaled_df)

# 设置因变量
y = df_clean[tobinq]  # 或其他目标变量

# 拟合OLS模型
model = sm.OLS(y, X_scaled_df).fit()

# 打印结果
print(model.summary())


print('-'*100)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 计算每个变量的 VIF
vif_data = pd.DataFrame()
vif_data["variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 输出 VIF 结果
print("\nVIF 分析结果：")
print(vif_data)