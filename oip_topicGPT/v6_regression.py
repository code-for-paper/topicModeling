import pandas as pd
import statsmodels.api as sm
import numpy as np

# Read your prepared table
df = pd.read_csv('./data/output/regression_input_example.csv')

tobinq = 'refer_tobins_q'
tobinq = 'tobinq'

# df['Supplier Collaboration_xrdint'] = df['Supplier Collaboration'] * df['xrdint']
# df['Technology Partnerships_xrdint'] = df['Technology Partnerships'] * df['xrdint']
# OIP topic variable squared terms (select a few)
# df['Supplier Collaboration_sq'] = df['Supplier Collaboration'] ** 2
# df['Technology Partnerships_sq'] = df['Technology Partnerships'] ** 2
# df['IP Licensing_sq'] = df['IP Licensing'] ** 2
# df['Strategic Partnerships_sq'] = df['Strategic Partnerships'] ** 2


# Set explanatory variables (including control variables and OIP topics)

explanatory_fixed_vars = [
    'log_at',
    'xrdint',
    'roa',
    'leverage', 
]
# ============================ Modify regression analysis items here =====================================
explanatory_topic_vars = [
    "Supplier & Manufacturing Collaboration",
    "Customer & Community Engagement",
    "Strategic & Joint Partnerships",
    "IP & Technology Licensing",
    "Industry-Academia & Research Collaboration",
    "Technology & Platform Ecosystems",
    "Open Innovation & Ecosystem Strategy",
    # "Marketing & Commercialization Cooperation",
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
    # "Marketing & Commercialization Cooperation_sq",
    # 'OIP_sq'
]



# Calculate interaction terms (OIP × xrdint)
for topic in explanatory_topic_vars:
    interaction_col = f"{topic}_xrdint"
    df[interaction_col] = df[topic] * df['xrdint']

# Calculate interaction terms (OIP × roa)
for topic in explanatory_topic_vars:
    interaction_col = f"{topic}_roa"
    df[interaction_col] = df[topic] * df['roa']

# Calculate squared terms (OIP ^ 2)
for topic in explanatory_topic_vars:
    quadratic_col = f"{topic}_sq"
    df[quadratic_col] = df[topic] ** 2


explanatory_vars = explanatory_fixed_vars + explanatory_topic_vars + other_vars

# Data cleaning: handle infinite and NaN values
print(f"Original data rows: {len(df)}")

# Check missing and infinite values for each column
for col in [tobinq] + explanatory_vars:
    if col in df.columns:
        inf_count = np.isinf(df[col]).sum()
        nan_count = df[col].isna().sum()
        if inf_count > 0 or nan_count > 0:
            print(f"{col}: {inf_count} infinite values, {nan_count} NaN values")

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Remove rows containing NaN values
df_clean = df[[tobinq] + explanatory_vars].dropna()
# ===================== Remove tail effects ====================
q_low = df_clean[tobinq].quantile(0.02)
q_high = df_clean[tobinq].quantile(0.99)

# Keep samples within normal range
df_clean = df_clean[(df_clean[tobinq] >= q_low) & (df_clean[tobinq] <= q_high)]

print(f"Cleaned data rows: {len(df_clean)}")
print(f"Removed {len(df) - len(df_clean)} rows of data")

# Prepare data
X = df_clean[explanatory_vars]
X = sm.add_constant(X)  # Add intercept term
y = df_clean[tobinq]         # Dependent variable

# Final check: ensure no inf or NaN
if np.any(np.isinf(X)) or np.any(np.isnan(X)) or np.any(np.isinf(y)) or np.any(np.isnan(y)):
    print("Warning: data still contains inf or NaN values")
else:
    print("Data cleaning complete, ready for regression analysis")

# Descriptive statistical analysis
print("Descriptive statistical analysis (dependent and explanatory variables):")
print(df_clean[[tobinq] + explanatory_vars].describe())

# Save descriptive statistics to csv file
desc_stats = df_clean[[tobinq] + explanatory_vars].describe()
desc_stats.to_csv('./data/output/desc_stats.csv', encoding='utf-8-sig')

print("Descriptive statistics results saved to './data/output/desc_stats.csv'")

# Calculate linear correlation coefficients between variables
print("\nLinear correlation coefficient matrix between variables:")
corr_matrix = df_clean[[tobinq] + explanatory_vars].corr()
print(corr_matrix)

# Significance test two-tailed 0.05
from scipy.stats import pearsonr

# 👇 Force entire matrix to string type (avoid dtype conflicts)
corr_matrix = corr_matrix.astype(str)

# Replace content with significance-marked strings, save only lower triangle
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        row_name = corr_matrix.index[i]
        col_name = corr_matrix.columns[j]
        
        if i == j:
            # Diagonal elements
            corr_matrix.loc[row_name, col_name] = "1.000"
        elif i > j:
            # Lower triangle elements
            r, p = pearsonr(df_clean[row_name], df_clean[col_name])
            if p < 0.05:
                corr_matrix.loc[row_name, col_name] = f"{r:.3f}*"
            else:
                corr_matrix.loc[row_name, col_name] = f"{r:.3f}"
        else:
            # Upper triangle elements set to empty string
            corr_matrix.loc[row_name, col_name] = ""

# Save correlation coefficient matrix to csv file
corr_matrix.to_csv('./data/output/correlation_matrix.csv', encoding='utf-8-sig')
print("Correlation coefficient matrix saved to './data/output/correlation_matrix.csv'")

print('-' * 100)



from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pandas as pd

# Extract and standardize explanatory variables
X = df_clean[explanatory_vars]
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)

# Convert standardized array to DataFrame and keep column names
X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

# Add constant term
X_scaled_df = sm.add_constant(X_scaled_df)

# Set dependent variable
y = df_clean[tobinq]  # Or other target variables

# Fit OLS model
model = sm.OLS(y, X_scaled_df).fit()

# Print results
print(model.summary())


print('-'*100)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data["variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Output VIF results
print("\nVIF analysis results:")
print(vif_data)