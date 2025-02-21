import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("preprocessed_data.csv")
#Handling Outliers

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(12, 6))
df[numeric_cols].boxplot()
plt.xticks(rotation=90)
plt.show()

# Capping outliers replaces extreme values with upper and lower percentile limits.
def cap_outliers(df, column, lower_percentile=0.05, upper_percentile=0.95):
    lower_limit = df[column].quantile(lower_percentile)
    upper_limit = df[column].quantile(upper_percentile)
    df[column] = np.where(df[column] < lower_limit, lower_limit, df[column])
    df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
for col in numeric_cols:
    cap_outliers(df, col)

plt.figure(figsize=(12, 6))
df[numeric_cols].boxplot()
plt.xticks(rotation=90)
plt.show()

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col} After Capping")
    plt.show()

df.to_csv("preprocessed_data.csv",index=False)